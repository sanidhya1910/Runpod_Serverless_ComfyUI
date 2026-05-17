# Scripts

This folder contains small utility scripts used by the worker and by local development.

## `send_runpod_request.py`

Send a ComfyUI workflow to the RunPod endpoint `9kkxxchvgfhb7g`.

### Requirements

- `RUNPOD_API_KEY` in a `.env` file at the project root
- A workflow JSON file exported from ComfyUI
- `user_id` for every request

### Basic usage

```bash
python send_runpod_request.py --user-id atlas --workflow test_input.json
```

### Inject a URL into the workflow

Use this when your workflow has a node that loads an image from a URL.

```bash
python scripts/send_runpod_request.py --user-id atlas --workflow test_input.json --image-url https://example.com/image.png --image-node-id 42 --image-input-key url
```

### Inject a text prompt into the workflow

Use this when your workflow has a prompt node that should be set from the command line.

```bash
python send_runpod_request.py --user-id atlas --workflow test_input.json --prompt "a cinematic photo of a red fox in snow" --prompt-node-id 6 --prompt-input-key text
```

### Useful flags

- `--endpoint-id`: override the default RunPod endpoint id
- `--async`: use `/run` instead of `/runsync`
- `--base-url`: override the RunPod API base URL
- `--timeout`: change the request timeout
- `--api-key`: override the key loaded from `.env`

### What the script prints

The script prints:

- the generated `job_id`
- each output filename
- the output URL when the worker returns an S3-compatible URL
- the full JSON response

### Other scripts in this folder

- `comfy-manager-set-mode.sh`
- `comfy-node-install.sh`
- `update-readme-version.js`
