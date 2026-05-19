"""Resolve R2 S3 API URLs to browser-accessible public URLs."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict
from urllib.parse import urlparse

R2_S3_HOST_SUFFIX = ".r2.cloudflarestorage.com"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_r2_s3_api_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.netloc or "").lower()
    return host.endswith(R2_S3_HOST_SUFFIX) or ".r2.cloudflarestorage.com" in host


def dedupe_bucket_prefix(path: str, bucket: str) -> str:
    """Fix keys like output-media/output-media/users/... -> output-media/users/..."""

    if not bucket:
        return path

    prefix = f"{bucket}/"
    while path.startswith(prefix + prefix):
        path = path[len(prefix) :]
    return path


def resolve_output_media_url(url: str) -> str:
    """
    Map an R2 S3 API URL to a public URL.

    Set R2_PUBLIC_BASE_URL (required for conversion), e.g.:
      dev:  https://pub-xxxx.r2.dev
      prod: https://cdn.yourdomain.com

    Optional:
      R2_BUCKET_NAME=output-media
      R2_PUBLIC_STRIP_BUCKET=true  # if the public host serves objects without bucket prefix
    """

    if not url or not is_r2_s3_api_url(url):
        return url

    public_base = os.getenv("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not public_base:
        return url

    bucket = os.getenv("R2_BUCKET_NAME", "output-media").strip()
    path = urlparse(url).path.lstrip("/")
    path = dedupe_bucket_prefix(path, bucket)

    if _env_bool("R2_PUBLIC_STRIP_BUCKET") and bucket and path.startswith(f"{bucket}/"):
        path = path[len(bucket) + 1 :]

    return f"{public_base}/{path}"


def resolve_output_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("type") != "s3_url":
        return item

    data = item.get("data")
    if not isinstance(data, str) or not data:
        return item

    public_url = resolve_output_media_url(data)
    if public_url == data:
        return item

    resolved = dict(item)
    resolved["s3_api_url"] = data
    resolved["data"] = public_url
    return resolved


def resolve_response_media_urls(response: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the RunPod response with public URLs in image/video items."""

    resolved = copy.deepcopy(response)
    output = resolved.get("output")
    if not isinstance(output, dict):
        return resolved

    for media_key in ("images", "videos"):
        items = output.get(media_key)
        if not isinstance(items, list):
            continue
        output[media_key] = [
            resolve_output_item(item) if isinstance(item, dict) else item for item in items
        ]

    return resolved
