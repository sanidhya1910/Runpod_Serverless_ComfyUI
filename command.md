# Sending requests to the RunPod serverless ComfyUI endpoint

The script reads `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` from `.env`.
`--user-id` is required (the worker rejects requests without it).

## Z-Image Turbo (workflows/zit_t2i.json)

The workflow has a single `{{prompt}}` placeholder on node `57:27.text`.

```powershell
python send_runpod_request.py `
  --user-id alice `
  --workflow workflows/zit_t2i.json `
  --set prompt="a serene mountain landscape at sunset, highly detailed"
```

Equivalent shorthand (auto-detects the `{{prompt}}` placeholder):

```powershell
python send_runpod_request.py `
  --user-id alice `
  --workflow workflows/zit_t2i.json `
  --prompt "a serene mountain landscape at sunset, highly detailed"
```

## Async (don't block on the job)

```powershell
python send_runpod_request.py `
  --user-id alice `
  --workflow workflows/zit_t2i.json `
  --prompt "a red panda eating bamboo" `
  --async
```

## Inspect a workflow before sending

```powershell
python send_runpod_request.py --user-id alice --workflow workflows/zit_t2i.json --list-nodes
```

## Health check (no job, no cost)

```powershell
$env:RUNPOD_API_KEY = (Get-Content .env | Select-String '^RUNPOD_API_KEY' | ForEach-Object { $_ -replace '^RUNPOD_API_KEY=\s*"?([^"]*)"?\s*$','$1' })
$env:RUNPOD_ENDPOINT_ID = (Get-Content .env | Select-String '^RUNPOD_ENDPOINT_ID' | ForEach-Object { $_ -replace '^RUNPOD_ENDPOINT_ID=\s*(.*)\s*$','$1' })
curl.exe -s -H "Authorization: Bearer $env:RUNPOD_API_KEY" "https://api.runpod.ai/v2/$env:RUNPOD_ENDPOINT_ID/health"
```
