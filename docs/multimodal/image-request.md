# Send a local image to llama-server

This walkthrough sends a PNG or JPEG from the client computer to a vision model through `/v1/chat/completions`. It uses Python's standard library and a base64 data URI, so the server does not need access to the client's filesystem or an image-hosting service.

## Start a vision model

Install [llama.cpp](../install.md), or [build](../build.md) the `llama-server` target. Run:

```sh
llama-server -hf ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0 \
    --alias local-vision --host 127.0.0.1 --port 8080 -c 4096
```

For a source build, replace `llama-server` with `./build/bin/llama-server`. The first launch downloads the model and its matching multimodal projector. Wait until loading finishes. This small model is useful for testing requests; its descriptions can be inaccurate.

In another terminal, check readiness:

```sh
curl --fail http://127.0.0.1:8080/health
```

A ready server returns `{"status":"ok"}`. Check its model capabilities as well:

```sh
curl --fail http://127.0.0.1:8080/v1/models
```

The model entry should include `multimodal` in `capabilities`. A text-only model cannot interpret an image, even if it accepts ordinary chat requests.

## Send the image

Save this as `describe_image.py`:

```python
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

image_path = Path(sys.argv[1])
media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[image_path.suffix.lower()]
image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
payload = {
    "model": "local-vision",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image briefly."},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
        ],
    }],
    "max_tokens": 128,
    "temperature": 0,
    "stream": False,
}
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print(error.read().decode("utf-8"), file=sys.stderr)
    raise
print(result["choices"][0]["message"]["content"])
```

Run it with your own image:

```sh
python3 describe_image.py /path/to/image.png
```

The request's `content` is a list containing separate text and image parts. The data URI contains the image bytes; putting a filename into the text prompt does not send the image.

## Diagnose a failed request

| Symptom | What to check |
| --- | --- |
| Connection refused | Confirm the server is running and the client uses the same host and port. |
| HTTP 503 from `/health` | Wait for model loading to finish; inspect the server log if it does not finish. |
| Model has no `multimodal` capability | Use a supported vision model with its matching projector. Do not start it with `--no-mmproj`. |
| Image decoding error | Confirm the file is a valid PNG or JPEG, rather than a downloaded HTML error page. |
| Context-size error | Images also consume context. Increase `-c` within your memory budget, or reduce the image size. |
| Plausible but incorrect description | A successful HTTP request confirms the inference path, not the accuracy of the model. Try a larger supported model and a clear image. |

The script prints the server's error body for HTTP failures. See the [server API reference](../../tools/server/README.md) for other image-input forms. A `file://` URL refers to a file accessible to the server and requires `--media-path`; it is different from uploading the client's bytes in a data URI.
