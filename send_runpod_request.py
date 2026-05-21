"""Helper for sending ComfyUI workflows to a RunPod serverless endpoint.

Usage:
    python scripts/send_runpod_request.py --workflow test_input.json

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import requests
from dotenv import load_dotenv

from media_urls import resolve_output_media_url, resolve_response_media_urls
from workflow_fields import (
    apply_field_values,
    build_placeholder_map,
    find_unfilled_placeholders,
)


DEFAULT_BASE_URL = "https://api.runpod.ai/v2"
DEFAULT_BINDINGS_PATH = Path(__file__).resolve().parent / "workflows" / "runpod_bindings.json"

# class_type -> default input field for user-facing prompt text
PROMPT_INPUT_KEYS: Dict[str, str] = {
    "CLIPTextEncode": "text",
    "CLIPTextEncodeSDXL": "text",
    "CLIPTextEncodeFlux": "text",
    "PrimitiveStringMultiline": "value",
    "StringConstant": "string",
    "Text": "text",
    "Text Multiline": "text",
}

# class_type -> default input field for image URLs
IMAGE_URL_INPUT_KEYS: Dict[str, str] = {
    "LoadImageFromUrl": "url",
    "LoadImageFromURL": "url",
    "ImageFromURL": "url",
    "LoadImageURL": "url",
}


load_dotenv()
DEFAULT_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")


def send_runpod_request(
    workflow: Dict[str, Any],
    user_id: str,
    api_key: str,
    endpoint_id: str = DEFAULT_ENDPOINT_ID,
    run_sync: bool = True,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Send a workflow to a RunPod ComfyUI endpoint."""

    route = "runsync" if run_sync else "run"
    url = f"{base_url.rstrip('/')}/{endpoint_id}/{route}"

    payload = {
        "input": {
            "user_id": user_id,
            "workflow": workflow,
        }
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def load_workflow(path: str) -> Dict[str, Any]:
    """Load a workflow JSON file from disk."""

    workflow_path = Path(path)
    with workflow_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return normalize_workflow(data)


def normalize_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the API workflow dict (unwrap RunPod/Comfy export wrappers)."""

    if isinstance(data.get("workflow"), dict):
        return data["workflow"]

    input_block = data.get("input")
    if isinstance(input_block, dict) and isinstance(input_block.get("workflow"), dict):
        return input_block["workflow"]

    return data


def load_bindings(path: Path | None = None) -> Dict[str, Any]:
    """Load optional per-workflow node bindings from JSON."""

    bindings_path = path or DEFAULT_BINDINGS_PATH
    if not bindings_path.is_file():
        return {}

    with bindings_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data if isinstance(data, dict) else {}


def workflow_binding_key(workflow_path: str) -> str:
    return Path(workflow_path).name


def get_embedded_binding(workflow: Dict[str, Any], role: str) -> Dict[str, str] | None:
    runpod_meta = workflow.get("_runpod")
    if not isinstance(runpod_meta, dict):
        return None
    binding = runpod_meta.get(role)
    if not isinstance(binding, dict):
        return None
    node_id = binding.get("node_id") or binding.get("node")
    input_key = binding.get("input_key") or binding.get("key")
    if isinstance(node_id, str) and isinstance(input_key, str):
        return {"node_id": node_id, "input_key": input_key}
    return None


def lookup_file_binding(
    bindings: Dict[str, Any], workflow_path: str, role: str
) -> Dict[str, str] | None:
    entry = bindings.get(workflow_binding_key(workflow_path))
    if not isinstance(entry, dict):
        return None
    binding = entry.get(role)
    if not isinstance(binding, dict):
        return None
    node_id = binding.get("node_id") or binding.get("node")
    input_key = binding.get("input_key") or binding.get("key")
    if isinstance(node_id, str) and isinstance(input_key, str):
        return {"node_id": node_id, "input_key": input_key}
    return None


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


def score_prompt_node(node_id: str, node: Dict[str, Any]) -> int | None:
    class_type = node.get("class_type")
    if not isinstance(class_type, str):
        return None

    input_key = PROMPT_INPUT_KEYS.get(class_type)
    if not input_key:
        return None

    title_lower = node_title(node).lower()
    if "negative" in title_lower:
        return None

    score = 0
    if "positive" in title_lower or "prompt" in title_lower:
        score += 100
    if class_type == "CLIPTextEncode":
        score += 50
    elif class_type == "PrimitiveStringMultiline":
        score += 45

    current = node.get("inputs", {}).get(input_key)
    if isinstance(current, str) and len(current.strip()) > 20:
        score += 15

    # Prefer lower ids when scores tie (stable, matches typical Comfy exports).
    score -= int(node_id) if node_id.isdigit() else 0
    return score


def find_prompt_candidates(workflow: Dict[str, Any]) -> list[Tuple[int, str, str, str]]:
    candidates: list[Tuple[int, str, str, str]] = []
    for node_id, node in iter_workflow_nodes(workflow):
        score = score_prompt_node(node_id, node)
        if score is None:
            continue
        class_type = node["class_type"]
        input_key = PROMPT_INPUT_KEYS[class_type]
        candidates.append((score, node_id, input_key, node_title(node)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def resolve_prompt_target(
    workflow: Dict[str, Any],
    workflow_path: str,
    *,
    node_id: str | None = None,
    input_key: str | None = None,
    node_title_query: str | None = None,
    bindings: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    if node_id:
        if input_key:
            return node_id, input_key
        node = workflow.get(node_id)
        if isinstance(node, dict):
            class_type = node.get("class_type", "")
            if isinstance(class_type, str) and class_type in PROMPT_INPUT_KEYS:
                return node_id, PROMPT_INPUT_KEYS[class_type]
        return node_id, "text"

    embedded = get_embedded_binding(workflow, "prompt")
    if embedded:
        return embedded["node_id"], embedded["input_key"]

    file_binding = lookup_file_binding(bindings or {}, workflow_path, "prompt")
    if file_binding:
        return file_binding["node_id"], file_binding["input_key"]

    if node_title_query:
        query = node_title_query.lower()
        matches = [
            (nid, node)
            for nid, node in iter_workflow_nodes(workflow)
            if query in node_title(node).lower()
        ]
        if len(matches) == 1:
            nid, node = matches[0]
            class_type = node.get("class_type", "")
            key = input_key or PROMPT_INPUT_KEYS.get(class_type, "text")
            return nid, key
        if len(matches) > 1:
            titles = ", ".join(f"{nid} ({node_title(n)})" for nid, n in matches)
            raise SystemExit(
                f"Multiple nodes match --prompt-node-title '{node_title_query}': {titles}"
            )
        raise SystemExit(f"No node title matches --prompt-node-title '{node_title_query}'.")

    candidates = find_prompt_candidates(workflow)
    if not candidates:
        raise SystemExit(
            "Could not find a prompt node. Use --prompt-node-id, --prompt-node-title, "
            "add workflows/runpod_bindings.json, or embed _runpod.prompt in the workflow."
        )
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        options = ", ".join(f"{nid} ({title or 'untitled'})" for _, nid, _, title in candidates[:5])
        raise SystemExit(
            "Ambiguous prompt node. Specify --prompt-node-id or --prompt-node-title. "
            f"Candidates: {options}"
        )

    _, best_id, best_key, best_title = candidates[0]
    print(f"auto prompt: node {best_id} ({best_title or 'untitled'}) -> inputs.{best_key}")
    return best_id, best_key


def score_image_url_node(node_id: str, node: Dict[str, Any]) -> int | None:
    class_type = node.get("class_type")
    if not isinstance(class_type, str):
        return None

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None

    if class_type in IMAGE_URL_INPUT_KEYS:
        return 80 - (int(node_id) if node_id.isdigit() else 0)

    if "url" in inputs:
        return 40 - (int(node_id) if node_id.isdigit() else 0)

    return None


def resolve_image_target(
    workflow: Dict[str, Any],
    workflow_path: str,
    *,
    node_id: str | None = None,
    input_key: str | None = None,
    bindings: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    if node_id:
        return node_id, input_key or "url"

    embedded = get_embedded_binding(workflow, "image")
    if embedded:
        return embedded["node_id"], embedded["input_key"]

    file_binding = lookup_file_binding(bindings or {}, workflow_path, "image")
    if file_binding:
        return file_binding["node_id"], file_binding["input_key"]

    scored: list[Tuple[int, str, str]] = []
    for nid, node in iter_workflow_nodes(workflow):
        score = score_image_url_node(nid, node)
        if score is None:
            continue
        class_type = node.get("class_type", "")
        key = IMAGE_URL_INPUT_KEYS.get(class_type, "url")
        scored.append((score, nid, key))
    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        raise SystemExit(
            "Could not find an image URL node. Use --image-node-id or add an image binding."
        )
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        options = ", ".join(nid for _, nid, _ in scored[:5])
        raise SystemExit(f"Ambiguous image URL node. Specify --image-node-id. Candidates: {options}")

    _, best_id, best_key = scored[0]
    print(f"auto image: node {best_id} -> inputs.{best_key}")
    return best_id, best_key


def list_workflow_nodes(workflow_path: str, bindings: Dict[str, Any]) -> None:
    workflow = load_workflow(workflow_path)
    print(f"Workflow: {workflow_path}")
    for node_id, node in sorted(iter_workflow_nodes(workflow), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        class_type = node.get("class_type", "?")
        title = node_title(node) or "(no title)"
        inputs = node.get("inputs")
        input_keys = ", ".join(inputs.keys()) if isinstance(inputs, dict) else ""
        markers: list[str] = []
        if score_prompt_node(node_id, node) is not None:
            markers.append("prompt?")
        if score_image_url_node(node_id, node) is not None:
            markers.append("image-url?")
        flag = f" [{', '.join(markers)}]" if markers else ""
        print(f"  {node_id}: {class_type} — {title} (inputs: {input_keys}){flag}")

    binding = lookup_file_binding(bindings, workflow_path, "prompt")
    if binding:
        print(f"  binding prompt -> {binding['node_id']}.{binding['input_key']}")
    embedded = get_embedded_binding(workflow, "prompt")
    if embedded:
        print(f"  embedded prompt -> {embedded['node_id']}.{embedded['input_key']}")


def inject_workflow_value(workflow: Dict[str, Any], node_id: str, input_key: str, value: str) -> None:
    """Inject a value into a specific workflow node input."""

    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise SystemExit(f"Workflow node '{node_id}' was not found.")

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit(f"Workflow node '{node_id}' does not contain an 'inputs' object.")

    inputs[input_key] = value


def extract_job_id(response: Dict[str, Any]) -> str | None:
    """Extract the job id from a RunPod response if present."""

    for key in ("id", "jobId", "job_id"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value

    output = response.get("output")
    if isinstance(output, dict):
        for key in ("id", "jobId", "job_id"):
            value = output.get(key)
            if isinstance(value, str) and value:
                return value

    return None


def iter_output_items(response: Dict[str, Any]):
    """Yield output items from the response in a predictable order."""

    output = response.get("output")
    if not isinstance(output, dict):
        return

    for output_type in ("images", "videos"):
        items = output.get(output_type)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield output_type, item


def print_response_summary(response: Dict[str, Any]) -> None:
    """Print the job id and any generated output files/URLs."""

    job_id = extract_job_id(response)
    if job_id:
        print(f"job_id: {job_id}")

    found_output = False
    for output_type, item in iter_output_items(response):
        found_output = True
        filename = item.get("filename", "<unknown>")
        data = item.get("data", "")
        item_type = item.get("type", "<unknown>")
        if item_type == "s3_url":
            display_url = resolve_output_media_url(data) if isinstance(data, str) else data
            print(f"{output_type[:-1]}: {filename} -> {display_url}")
        else:
            print(f"{output_type[:-1]}: {filename} ({item_type})")

    if not found_output:
        print("No generated outputs were included in the response.")


def parse_set_args(set_args: list[str] | None) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for item in set_args or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --set '{item}'. Use NAME=VALUE (e.g. --set positive_prompt=hello).")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid --set '{item}'. Name cannot be empty.")
        values[name] = value
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a ComfyUI workflow to RunPod")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key (defaults to RUNPOD_API_KEY from .env)")
    parser.add_argument("--endpoint-id", default=DEFAULT_ENDPOINT_ID, help="RunPod endpoint id (defaults to RUNPOD_ENDPOINT_ID from .env)")
    parser.add_argument("--user-id", required=True, help="Required user id for the job")
    parser.add_argument("--workflow", required=True, help="Path to a workflow JSON file")
    parser.add_argument("--image-url", help="Optional URL to inject into a workflow node before sending")
    parser.add_argument("--image-node-id", help="Workflow node id to receive the URL")
    parser.add_argument("--image-input-key", default="url", help="Input key to set on the workflow node (default: url)")
    parser.add_argument("--prompt", help="Optional text prompt to inject into a workflow node before sending")
    parser.add_argument(
        "--prompt-node-id",
        help="Workflow node id for the prompt (optional; auto-detected if omitted)",
    )
    parser.add_argument(
        "--prompt-input-key",
        help="Input key on the prompt node (default: auto from node type, else text)",
    )
    parser.add_argument(
        "--prompt-node-title",
        help="Pick prompt node by partial match on ComfyUI _meta.title (e.g. 'Positive')",
    )
    parser.add_argument(
        "--bindings",
        default=str(DEFAULT_BINDINGS_PATH),
        help=f"Optional JSON map of workflow bindings (default: {DEFAULT_BINDINGS_PATH.name})",
    )
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="Print workflow nodes and likely prompt/image targets, then exit",
    )
    parser.add_argument("--async", dest="run_sync", action="store_false", help="Use /run instead of /runsync")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="RunPod API base URL")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument(
        "--raw-urls",
        action="store_true",
        help="Do not rewrite R2 S3 API URLs to R2_PUBLIC_BASE_URL",
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        help="Set a {{placeholder}} or node.input value (repeatable)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.api_key:
        raise SystemExit("Missing RunPod API key. Add RUNPOD_API_KEY to .env or pass --api-key.")

    bindings_path = Path(args.bindings)
    bindings = load_bindings(bindings_path if bindings_path.is_file() else None)

    if args.list_nodes:
        list_workflow_nodes(args.workflow, bindings)
        return

    workflow = load_workflow(args.workflow)

    set_values = parse_set_args(args.set)
    if set_values:
        workflow = apply_field_values(workflow, set_values)

    if args.image_url:
        image_node_id, image_input_key = resolve_image_target(
            workflow,
            args.workflow,
            node_id=args.image_node_id,
            input_key=args.image_input_key,
            bindings=bindings,
        )
        inject_workflow_value(workflow, image_node_id, image_input_key, args.image_url)

    if args.prompt:
        placeholders = build_placeholder_map(workflow)
        if args.prompt_node_id:
            input_key = args.prompt_input_key or "text"
            inject_workflow_value(workflow, args.prompt_node_id, input_key, args.prompt)
        elif placeholders:
            name = next(
                (n for n in ("positive_prompt", "prompt") if n in placeholders),
                sorted(placeholders.keys())[0],
            )
            node_id, input_key = placeholders[name]
            inject_workflow_value(workflow, node_id, input_key, args.prompt)
        else:
            prompt_node_id, prompt_input_key = resolve_prompt_target(
                workflow,
                args.workflow,
                node_id=None,
                input_key=args.prompt_input_key,
                node_title_query=args.prompt_node_title,
                bindings=bindings,
            )
            inject_workflow_value(workflow, prompt_node_id, prompt_input_key, args.prompt)

    unfilled = find_unfilled_placeholders(workflow)
    if unfilled:
        names = ", ".join(unfilled)
        raise SystemExit(
            f"Workflow still has unfilled placeholders: {names}. "
            f"Use --set name=value for each (e.g. --set {unfilled[0]}=...)."
        )

    result = send_runpod_request(
        workflow=workflow,
        user_id=args.user_id,
        api_key=args.api_key,
        endpoint_id=args.endpoint_id,
        run_sync=args.run_sync,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    if not args.raw_urls:
        result = resolve_response_media_urls(result)
    print_response_summary(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()