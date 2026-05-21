"""Discover and apply user-editable fields in ComfyUI API workflows."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple

# Value must be exactly {{name}} — marks an input to replace before sending to RunPod.
PLACEHOLDER_PATTERN = re.compile(r"^\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}$")

WidgetKind = Literal[
    "text",
    "text_short",
    "number",
    "slider",
    "dropdown",
    "checkbox",
    "image_url",
    "image_path",
]

PROMPT_INPUT_KEYS: Dict[str, str] = {
    "CLIPTextEncode": "text",
    "CLIPTextEncodeSDXL": "text",
    "CLIPTextEncodeFlux": "text",
    "PrimitiveStringMultiline": "value",
    "StringConstant": "string",
    "Text": "text",
    "Text Multiline": "text",
}

IMAGE_URL_INPUT_KEYS: Dict[str, str] = {
    "LoadImageFromUrl": "url",
    "LoadImageFromURL": "url",
    "ImageFromURL": "url",
    "LoadImageURL": "url",
}

# Inputs hidden unless listed in _runpod.ui or bindings "ui"
SKIP_INPUT_KEYS = frozenset(
    {
        "filename_prefix",
        "ckpt_name",
        "vae_name",
        "model_name",
        "lora_name",
        "clip_name",
        "control_net_name",
    }
)

ENUM_CHOICES: Dict[str, list[str]] = {
    "image_size": [
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
    ],
    "aspect_ratio": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
    "sampler_name": [
        "euler",
        "euler_ancestral",
        "heun",
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpm_fast",
        "dpm_adaptive",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_2m",
        "ddim",
        "uni_pc",
    ],
    "scheduler": ["simple", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform"],
}

NUMBER_HINTS: Dict[str, Tuple[float | None, float | None, float | None]] = {
    "width": (256, 2048, 64),
    "height": (256, 2048, 64),
    "steps": (1, 150, 1),
    "num_inference_steps": (1, 150, 1),
    "cfg": (0.0, 30.0, 0.5),
    "guidance": (0.0, 30.0, 0.5),
    "guidance_scale": (0.0, 30.0, 0.5),
    "denoise": (0.0, 1.0, 0.05),
    "seed": (-1, 2**53 - 1, 1),
    "batch_size": (1, 16, 1),
    "num_images": (1, 8, 1),
    "duration": (1, 60, 1),
    "length": (1, 600, 1),
    "seconds": (1, 120, 1),
    "num_frames": (1, 256, 1),
    "frame_count": (1, 256, 1),
    "video_length": (1, 256, 1),
    "fps": (1, 60, 1),
}


@dataclass
class FieldSpec:
    field_id: str
    node_id: str
    input_key: str
    label: str
    widget: WidgetKind
    default: Any
    group: str = "Parameters"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] = field(default_factory=list)
    lines: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_id": self.field_id,
            "node_id": self.node_id,
            "input_key": self.input_key,
            "label": self.label,
            "widget": self.widget,
            "default": self.default,
            "group": self.group,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": self.choices,
            "lines": self.lines,
        }


def normalize_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data.get("workflow"), dict):
        return data["workflow"]
    input_block = data.get("input")
    if isinstance(input_block, dict) and isinstance(input_block.get("workflow"), dict):
        return input_block["workflow"]
    return data


def node_title(node: Dict[str, Any]) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict) and isinstance(meta.get("title"), str):
        return meta["title"]
    return ""


def iter_workflow_nodes(workflow: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for node_id, node in workflow.items():
        if node_id.startswith("_"):
            continue
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        yield str(node_id), node


def is_linked_value(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    slot = value[1]
    return isinstance(slot, int)


def field_id(node_id: str, input_key: str) -> str:
    return f"{node_id}.{input_key}"


def parse_field_id(field_id_str: str) -> Tuple[str, str]:
    node_id, _, input_key = field_id_str.partition(".")
    if not node_id or not input_key:
        raise ValueError(f"Invalid field id: {field_id_str}")
    return node_id, input_key


def parse_placeholder_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = PLACEHOLDER_PATTERN.match(value.strip())
    return match.group(1) if match else None


def iter_placeholders_in_workflow(
    workflow: Dict[str, Any],
) -> Iterable[Tuple[str, str, str, Dict[str, Any]]]:
    """Yield (placeholder_name, node_id, input_key, node) for each {{name}} input."""

    for node_id, node in iter_workflow_nodes(workflow):
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_key, value in inputs.items():
            if is_linked_value(value):
                continue
            name = parse_placeholder_name(value)
            if name:
                yield name, node_id, input_key, node


def build_placeholder_map(workflow: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Map placeholder name -> (node_id, input_key)."""

    mapping: Dict[str, Tuple[str, str]] = {}
    for name, node_id, input_key, _node in iter_placeholders_in_workflow(workflow):
        if name in mapping and mapping[name] != (node_id, input_key):
            raise ValueError(
                f"Placeholder '{{{{{name}}}}}' is used on multiple inputs: "
                f"{mapping[name]} and {(node_id, input_key)}"
            )
        mapping[name] = (node_id, input_key)
    return mapping


def resolve_field_target(workflow: Dict[str, Any], field_key: str) -> Tuple[str, str]:
    """Resolve field id: either 'node.input' or a {{placeholder}} name."""

    if "." in field_key:
        return parse_field_id(field_key)
    mapping = build_placeholder_map(workflow)
    if field_key in mapping:
        return mapping[field_key]
    raise ValueError(f"Unknown field '{field_key}' (not node.input or {{placeholder}})")


def find_unfilled_placeholders(workflow: Dict[str, Any]) -> list[str]:
    unfilled: list[str] = []
    for name, _node_id, _input_key, _node in iter_placeholders_in_workflow(workflow):
        unfilled.append(name)
    return unfilled


def classify_string_field(
    node_id: str,
    node: Dict[str, Any],
    input_key: str,
    value: str,
) -> FieldSpec | None:
    class_type = node.get("class_type", "")
    title = node_title(node)
    label_base = title or f"{class_type} ({node_id})"
    label = f"{label_base} — {input_key}"

    if class_type in IMAGE_URL_INPUT_KEYS and input_key == IMAGE_URL_INPUT_KEYS[class_type]:
        return FieldSpec(
            field_id=field_id(node_id, input_key),
            node_id=node_id,
            input_key=input_key,
            label=label,
            widget="image_url",
            default=value,
            group="Images",
        )

    if input_key == "url" or input_key.endswith("_url"):
        return FieldSpec(
            field_id=field_id(node_id, input_key),
            node_id=node_id,
            input_key=input_key,
            label=label,
            widget="image_url",
            default=value,
            group="Images",
        )

    if class_type == "LoadImage" and input_key == "image":
        return FieldSpec(
            field_id=field_id(node_id, input_key),
            node_id=node_id,
            input_key=input_key,
            label=label,
            widget="image_path",
            default=value,
            group="Images",
        )

    prompt_key = PROMPT_INPUT_KEYS.get(class_type)
    if prompt_key == input_key or input_key in ("text", "prompt", "positive", "negative"):
        group = "Prompts"
        if "negative" in title.lower() or input_key == "negative":
            group = "Prompts (negative)"
        lines = 6 if len(value) > 80 or input_key in ("text", "value") else 2
        return FieldSpec(
            field_id=field_id(node_id, input_key),
            node_id=node_id,
            input_key=input_key,
            label=label,
            widget="text",
            default=value,
            group=group,
            lines=lines,
        )

    if input_key in SKIP_INPUT_KEYS:
        return None

    if input_key in ENUM_CHOICES:
        choices = list(ENUM_CHOICES[input_key])
        if value not in choices:
            choices.insert(0, value)
        return FieldSpec(
            field_id=field_id(node_id, input_key),
            node_id=node_id,
            input_key=input_key,
            label=label,
            widget="dropdown",
            default=value,
            choices=choices,
            group="Parameters",
        )

    if len(value) > 64:
        return FieldSpec(
            field_id=field_id(node_id, input_key),
            node_id=node_id,
            input_key=input_key,
            label=label,
            widget="text",
            default=value,
            group="Parameters",
            lines=4,
        )

    return FieldSpec(
        field_id=field_id(node_id, input_key),
        node_id=node_id,
        input_key=input_key,
        label=label,
        widget="text_short",
        default=value,
        group="Parameters",
    )


def classify_numeric_field(
    node_id: str,
    node: Dict[str, Any],
    input_key: str,
    value: int | float,
) -> FieldSpec:
    class_type = node.get("class_type", "")
    title = node_title(node)
    label = f"{title or class_type} ({node_id}) — {input_key}"

    hints = NUMBER_HINTS.get(input_key)
    min_v, max_v, step_v = hints if hints else (None, None, None)

    if input_key in ("duration", "length", "seconds", "num_frames", "frame_count", "video_length", "fps"):
        group = "Video"
    elif input_key in ("width", "height", "batch_size"):
        group = "Resolution"
    elif input_key in ("steps", "num_inference_steps", "cfg", "guidance", "guidance_scale", "denoise", "seed"):
        group = "Sampling"
    else:
        group = "Parameters"

    widget: WidgetKind = "slider" if hints else "number"
    if isinstance(value, float) and not step_v:
        step_v = 0.01

    return FieldSpec(
        field_id=field_id(node_id, input_key),
        node_id=node_id,
        input_key=input_key,
        label=label,
        widget=widget,
        default=value,
        group=group,
        min=min_v,
        max=max_v,
        step=step_v,
    )


def field_from_ui_entry(entry: Dict[str, Any], workflow: Dict[str, Any]) -> FieldSpec | None:
    node_id = str(entry.get("node_id") or entry.get("node", ""))
    input_key = str(entry.get("input_key") or entry.get("key", ""))
    if not node_id or not input_key:
        return None

    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return None

    inputs = node.get("inputs")
    default = entry.get("default")
    if default is None and isinstance(inputs, dict):
        default = inputs.get(input_key)

    widget = entry.get("widget", "text")
    label = entry.get("label") or f"{node_title(node) or node.get('class_type')} ({node_id}) — {input_key}"
    fid = entry.get("id") or field_id(node_id, input_key)

    return FieldSpec(
        field_id=str(fid),
        node_id=node_id,
        input_key=input_key,
        label=str(label),
        widget=widget,
        default=default if default is not None else "",
        group=str(entry.get("group", "Custom")),
        min=entry.get("min"),
        max=entry.get("max"),
        step=entry.get("step"),
        choices=list(entry.get("choices", [])),
        lines=int(entry.get("lines", 3)),
    )


def get_runpod_ui_config(workflow: Dict[str, Any]) -> Dict[str, Any]:
    meta = workflow.get("_runpod")
    return meta if isinstance(meta, dict) else {}


def placeholder_config(runpod: Dict[str, Any], name: str) -> Dict[str, Any]:
    placeholders = runpod.get("placeholders")
    if not isinstance(placeholders, dict):
        return {}
    entry = placeholders.get(name)
    return entry if isinstance(entry, dict) else {}


def field_from_placeholder(
    name: str,
    node_id: str,
    input_key: str,
    node: Dict[str, Any],
    runpod: Dict[str, Any],
) -> FieldSpec:
    config = placeholder_config(runpod, name)
    label = config.get("label") or f"{name.replace('_', ' ').title()} ({node_id}.{input_key})"
    group = str(config.get("group", "Inputs"))
    widget_override = config.get("widget")

    if widget_override:
        widget = widget_override
        default: Any = config.get("default", "")
        spec = FieldSpec(
            field_id=name,
            node_id=node_id,
            input_key=input_key,
            label=str(label),
            widget=widget,
            default=default,
            group=group,
            min=config.get("min"),
            max=config.get("max"),
            step=config.get("step"),
            choices=list(config.get("choices", [])),
            lines=int(config.get("lines", 3)),
        )
        return spec

    # Infer widget from node/input (treat placeholder default as empty string for prompts).
    inferred = classify_string_field(node_id, node, input_key, "")
    if inferred:
        inferred.field_id = name
        inferred.label = str(label)
        inferred.group = group
        inferred.default = config.get("default", "")
        return inferred

    hints = NUMBER_HINTS.get(input_key)
    if hints:
        return FieldSpec(
            field_id=name,
            node_id=node_id,
            input_key=input_key,
            label=str(label),
            widget="slider",
            default=config.get("default", hints[0] or 0),
            group=group,
            min=config.get("min", hints[0]),
            max=config.get("max", hints[1]),
            step=config.get("step", hints[2]),
        )

    return FieldSpec(
        field_id=name,
        node_id=node_id,
        input_key=input_key,
        label=str(label),
        widget="text_short",
        default=config.get("default", ""),
        group=group,
    )


def discover_placeholder_fields(workflow: Dict[str, Any]) -> list[FieldSpec]:
    runpod = get_runpod_ui_config(workflow)
    specs: list[FieldSpec] = []
    for name, node_id, input_key, node in iter_placeholders_in_workflow(workflow):
        specs.append(field_from_placeholder(name, node_id, input_key, node, runpod))
    specs.sort(key=lambda s: s.label)
    return specs


def discover_fields(
    workflow: Dict[str, Any],
    *,
    explicit_only: bool = False,
    ui_entries: list[Dict[str, Any]] | None = None,
) -> list[FieldSpec]:
    """Return editable fields for a workflow, grouped for UI display."""

    runpod = get_runpod_ui_config(workflow)
    entries = ui_entries if ui_entries is not None else runpod.get("ui")
    if isinstance(entries, list) and entries:
        specs = [field_from_ui_entry(e, workflow) for e in entries]
        return [s for s in specs if s is not None]

    placeholder_specs = discover_placeholder_fields(workflow)
    if placeholder_specs:
        return placeholder_specs

    if explicit_only:
        return []

    hide_set = set()
    for item in runpod.get("hide", []) if isinstance(runpod.get("hide"), list) else []:
        if isinstance(item, dict):
            hide_set.add(field_id(str(item.get("node_id", "")), str(item.get("input_key", ""))))

    specs: list[FieldSpec] = []
    seen: set[str] = set()

    for node_id, node in iter_workflow_nodes(workflow):
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        for input_key, value in inputs.items():
            if is_linked_value(value):
                continue

            fid = field_id(node_id, input_key)
            if fid in hide_set or fid in seen:
                continue

            spec: FieldSpec | None = None
            if isinstance(value, bool):
                title = node_title(node) or node.get("class_type", "")
                spec = FieldSpec(
                    field_id=fid,
                    node_id=node_id,
                    input_key=input_key,
                    label=f"{title} ({node_id}) — {input_key}",
                    widget="checkbox",
                    default=value,
                    group="Parameters",
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if input_key in SKIP_INPUT_KEYS:
                    continue
                spec = classify_numeric_field(node_id, node, input_key, value)
            elif isinstance(value, str):
                if input_key in SKIP_INPUT_KEYS and input_key not in PROMPT_INPUT_KEYS.values():
                    class_type = node.get("class_type", "")
                    if not (class_type in PROMPT_INPUT_KEYS and input_key == PROMPT_INPUT_KEYS[class_type]):
                        continue
                spec = classify_string_field(node_id, node, input_key, value)

            if spec:
                seen.add(fid)
                specs.append(spec)

    group_order = {
        "Prompts": 0,
        "Prompts (negative)": 1,
        "Images": 2,
        "Resolution": 3,
        "Sampling": 4,
        "Video": 5,
        "Parameters": 6,
        "Custom": 7,
    }
    specs.sort(key=lambda s: (group_order.get(s.group, 99), s.label))
    return specs


def load_workflow_file(path: str | Path) -> Dict[str, Any]:
    workflow_path = Path(path)
    with workflow_path.open("r", encoding="utf-8") as handle:
        return normalize_workflow(json.load(handle))


def load_bindings_ui(workflow_name: str, bindings: Dict[str, Any]) -> list[Dict[str, Any]] | None:
    entry = bindings.get(workflow_name)
    if not isinstance(entry, dict):
        return None
    ui = entry.get("ui")
    return ui if isinstance(ui, list) else None


def coerce_input_value(current: Any, raw_value: Any) -> Any:
    if isinstance(current, bool):
        return bool(raw_value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw_value))
    if isinstance(current, float):
        return float(raw_value)
    if parse_placeholder_name(current) is not None:
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            return int(raw_value) if float(raw_value).is_integer() else float(raw_value)
        return raw_value
    return raw_value


def apply_field_values(workflow: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the workflow with user values applied (by node.input or {{placeholder}} name)."""

    patched = copy.deepcopy(workflow)
    for fid, raw_value in values.items():
        if raw_value is None:
            continue
        if isinstance(raw_value, str) and not raw_value.strip() and "." in fid and fid.endswith(".text"):
            continue

        try:
            node_id, input_key = resolve_field_target(patched, fid)
        except ValueError:
            continue

        node = patched.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        current = inputs.get(input_key)
        inputs[input_key] = coerce_input_value(current, raw_value)

    return patched


def group_fields(specs: list[FieldSpec]) -> Dict[str, list[FieldSpec]]:
    groups: Dict[str, list[FieldSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.group, []).append(spec)
    return groups
