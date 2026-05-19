"""Gradio UI for editing ComfyUI workflow fields and sending jobs to RunPod."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from media_urls import resolve_output_media_url, resolve_response_media_urls
from send_runpod_request import (
    DEFAULT_BINDINGS_PATH,
    DEFAULT_ENDPOINT_ID,
    extract_job_id,
    iter_output_items,
    load_bindings,
    send_runpod_request,
)
from workflow_fields import (
    FieldSpec,
    apply_field_values,
    discover_fields,
    group_fields,
    load_bindings_ui,
    load_workflow_file,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent
WORKFLOWS_DIR = ROOT / "workflows"


def list_workflow_files() -> list[str]:
    if not WORKFLOWS_DIR.is_dir():
        return []
    return sorted(path.name for path in WORKFLOWS_DIR.glob("*.json"))


def resolve_workflow_path(name: str) -> Path:
    return WORKFLOWS_DIR / name


def load_specs_for_workflow(workflow_name: str) -> list[FieldSpec]:
    path = resolve_workflow_path(workflow_name)
    workflow = load_workflow_file(path)
    bindings = load_bindings(DEFAULT_BINDINGS_PATH if DEFAULT_BINDINGS_PATH.is_file() else None)
    ui_entries = load_bindings_ui(workflow_name, bindings)
    return discover_fields(workflow, ui_entries=ui_entries)


def specs_to_state(specs: list[FieldSpec]) -> list[dict[str, Any]]:
    return [s.to_dict() for s in specs]


def state_to_specs(state: list[dict[str, Any]]) -> list[FieldSpec]:
    return [FieldSpec(**item) for item in state]


def make_widget(spec: FieldSpec):
    if spec.widget == "text":
        return gr.Textbox(
            label=spec.label,
            value=spec.default if spec.default is not None else "",
            lines=spec.lines,
        )
    if spec.widget == "text_short":
        return gr.Textbox(label=spec.label, value=spec.default if spec.default is not None else "")
    if spec.widget == "image_url":
        return gr.Textbox(label=spec.label, placeholder="https://...", value=spec.default or "")
    if spec.widget == "image_path":
        return gr.Textbox(
            label=spec.label,
            info="Filename on the ComfyUI worker (input/ folder), or use a URL workflow node instead.",
            value=spec.default or "",
        )
    if spec.widget == "checkbox":
        return gr.Checkbox(label=spec.label, value=bool(spec.default))
    if spec.widget == "dropdown" and spec.choices:
        value = spec.default if spec.default in spec.choices else spec.choices[0]
        return gr.Dropdown(label=spec.label, choices=spec.choices, value=value)
    if spec.widget == "slider":
        return gr.Slider(
            label=spec.label,
            value=float(spec.default) if spec.default is not None else 0,
            minimum=spec.min if spec.min is not None else 0,
            maximum=spec.max if spec.max is not None else 100,
            step=spec.step if spec.step is not None else 1,
        )
    return gr.Number(
        label=spec.label,
        value=float(spec.default) if spec.default is not None else 0,
        minimum=spec.min,
        maximum=spec.max,
        step=spec.step or 1,
    )


def response_to_gallery(response: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for _output_type, item in iter_output_items(response):
        data = item.get("data", "")
        item_type = item.get("type", "")
        if item_type == "s3_url" and isinstance(data, str) and data:
            urls.append(resolve_output_media_url(data))
        elif isinstance(data, str) and data and item_type not in ("s3_url",):
            mime = "image/png" if _output_type == "images" else "video/mp4"
            urls.append(f"data:{mime};base64,{data}")
    return urls


def format_status(response: dict[str, Any]) -> str:
    lines: list[str] = []
    job_id = extract_job_id(response)
    if job_id:
        lines.append(f"Job ID: {job_id}")

    for output_type, item in iter_output_items(response):
        filename = item.get("filename", "output")
        data = item.get("data", "")
        item_type = item.get("type", "")
        if item_type == "s3_url":
            lines.append(f"{output_type[:-1]}: {filename}\n  {resolve_output_media_url(data)}")
        else:
            lines.append(f"{output_type[:-1]}: {filename} ({item_type})")

    if not lines:
        status = response.get("status", "")
        if status:
            lines.append(f"Status: {status}")
        else:
            lines.append("Job finished (see JSON for full response).")
    return "\n".join(lines)


def run_workflow(
    workflow_name: str,
    user_id: str,
    field_specs_state: list[dict[str, Any]],
    run_sync: bool,
    timeout: float,
    *field_values: Any,
) -> tuple[str, list[str], dict[str, Any]]:
    api_key = os.getenv("RUNPOD_API_KEY", "")
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID)
    if not api_key:
        return "Missing RUNPOD_API_KEY in .env", [], {}
    if not user_id.strip():
        return "User ID is required.", [], {}
    if not workflow_name:
        return "Select a workflow.", [], {}

    specs = state_to_specs(field_specs_state)
    if len(field_values) != len(specs):
        return (
            f"Field count mismatch ({len(field_values)} values, {len(specs)} specs). Reload the workflow.",
            [],
            {},
        )

    values_map = {spec.field_id: value for spec, value in zip(specs, field_values)}

    path = resolve_workflow_path(workflow_name)
    workflow = load_workflow_file(path)
    workflow = apply_field_values(workflow, values_map)

    try:
        response = send_runpod_request(
            workflow=workflow,
            user_id=user_id.strip(),
            api_key=api_key,
            endpoint_id=endpoint_id,
            run_sync=run_sync,
            timeout=timeout,
        )
    except Exception as exc:
        return f"Request failed: {exc}", [], {}

    response = resolve_response_media_urls(response)
    gallery = response_to_gallery(response)
    status = format_status(response)
    return status, gallery, response


def build_ui() -> gr.Blocks:
    workflow_choices = list_workflow_files()
    default_workflow = workflow_choices[0] if workflow_choices else None
    initial_specs = load_specs_for_workflow(default_workflow) if default_workflow else []

    with gr.Blocks(title="RunPod ComfyUI Workflow Editor") as demo:
        gr.Markdown(
            "## RunPod ComfyUI — Workflow Editor\n"
            "Pick a workflow, edit every exposed prompt, image, resolution, sampling, and video "
            "parameter, then send the job. Fields are discovered automatically from the workflow JSON; "
            "use `_runpod.ui` in the workflow or `ui` in `workflows/runpod_bindings.json` to customize."
        )

        with gr.Row():
            workflow_dd = gr.Dropdown(
                label="Workflow",
                choices=workflow_choices,
                value=default_workflow,
            )
            user_id_tb = gr.Textbox(label="User ID", value="atlas", placeholder="Required")
            run_sync_cb = gr.Checkbox(label="Wait for result (runsync)", value=True)
            timeout_nb = gr.Number(label="Timeout (seconds)", value=300, precision=0)

        specs_state = gr.State(specs_to_state(initial_specs))

        with gr.Row():
            reload_btn = gr.Button("Reload fields from workflow", variant="secondary")
            field_count_md = gr.Markdown(
                value=f"**{len(initial_specs)}** editable fields loaded."
            )

        @reload_btn.click(inputs=[workflow_dd], outputs=[specs_state, field_count_md])
        def reload_specs(workflow_name: str):
            specs = load_specs_for_workflow(workflow_name)
            return specs_to_state(specs), f"**{len(specs)}** editable fields loaded."

        @workflow_dd.change(inputs=[workflow_dd], outputs=[specs_state, field_count_md])
        def on_workflow_change(workflow_name: str):
            specs = load_specs_for_workflow(workflow_name)
            return specs_to_state(specs), f"**{len(specs)}** editable fields loaded."

        status_tb = gr.Textbox(label="Status", lines=6, interactive=False)
        gallery = gr.Gallery(label="Outputs", columns=2, height=400)
        response_json = gr.JSON(label="Raw response")

        @gr.render(inputs=[specs_state, user_id_tb, workflow_dd, run_sync_cb, timeout_nb])
        def render_fields(
            specs_state_value: list[dict[str, Any]],
            user_id: str,
            workflow_name: str,
            run_sync: bool,
            timeout: float,
        ):
            specs = state_to_specs(specs_state_value)
            if not specs:
                gr.Markdown("*No editable fields found. Add literals to the workflow or define `_runpod.ui`.*")
                return

            widgets: list[Any] = []
            grouped = group_fields(specs)
            for group_name, group_specs in grouped.items():
                with gr.Accordion(group_name, open=group_name in ("Prompts", "Images", "Resolution", "Video")):
                    for spec in group_specs:
                        widgets.append(make_widget(spec))

            gr.Markdown(f"*{len(specs)} parameters in this workflow*")
            run_btn = gr.Button("Run on RunPod", variant="primary")

            run_btn.click(
                fn=run_workflow,
                inputs=[workflow_dd, user_id_tb, specs_state, run_sync_cb, timeout_nb, *widgets],
                outputs=[status_tb, gallery, response_json],
            )

        with gr.Accordion("Customize field discovery", open=False):
            gr.Markdown(
                """
**Automatic discovery** includes:
- All **prompt** nodes (`CLIPTextEncode`, `PrimitiveStringMultiline`, …) — including multiple positive/negative prompts
- **Image** URL nodes and `LoadImage` filename inputs (one field per image node)
- **Resolution** (`width`, `height`, `batch_size`)
- **Sampling** (`seed`, `steps`, `cfg`, `guidance`, …)
- **Video** (`duration`, `num_frames`, `fps`, …)

**Hide** noisy fields by default: `filename_prefix`, `ckpt_name`, etc.

**Override** with explicit UI config in the workflow file:

```json
"_runpod": {
  "ui": [
    {"node_id": "6", "input_key": "text", "label": "Positive prompt", "widget": "text", "group": "Prompts"},
    {"node_id": "27", "input_key": "width", "widget": "slider", "min": 512, "max": 1536, "step": 64}
  ],
  "hide": [{"node_id": "9", "input_key": "filename_prefix"}]
}
```

Or in `workflows/runpod_bindings.json` per workflow:

```json
"my_workflow.json": {
  "ui": [ ... same entries ... ]
}
```
"""
            )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
