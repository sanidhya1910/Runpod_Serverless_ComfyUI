"""Helper for sending ComfyUI workflows to a RunPod serverless endpoint.

Usage:
    python scripts/send_runpod_request.py --workflow test_input.json

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.runpod.ai/v2"


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
        return json.load(file_handle)


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
            print(f"{output_type[:-1]}: {filename} -> {data}")
        else:
            print(f"{output_type[:-1]}: {filename} ({item_type})")

    if not found_output:
        print("No generated outputs were included in the response.")


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
    parser.add_argument("--prompt-node-id", help="Workflow node id to receive the prompt")
    parser.add_argument("--prompt-input-key", default="text", help="Input key to set on the workflow node (default: text)")
    parser.add_argument("--async", dest="run_sync", action="store_false", help="Use /run instead of /runsync")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="RunPod API base URL")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.api_key:
        raise SystemExit("Missing RunPod API key. Add RUNPOD_API_KEY to .env or pass --api-key.")

    workflow = load_workflow(args.workflow)
    if args.image_url:
        if not args.image_node_id:
            raise SystemExit("--image-node-id is required when using --image-url.")
        inject_workflow_value(workflow, args.image_node_id, args.image_input_key, args.image_url)

    if args.prompt:
        if not args.prompt_node_id:
            raise SystemExit("--prompt-node-id is required when using --prompt.")
        inject_workflow_value(workflow, args.prompt_node_id, args.prompt_input_key, args.prompt)

    result = send_runpod_request(
        workflow=workflow,
        user_id=args.user_id,
        api_key=args.api_key,
        endpoint_id=args.endpoint_id,
        run_sync=args.run_sync,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    print_response_summary(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()