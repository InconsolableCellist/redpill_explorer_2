import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from re2 import vlm

CFG = SimpleNamespace(base_url="http://vlm.test/v1", vision_model="m", vision_max_tokens=100)


def sample_result():
    return {
        "media_type": "meme", "description": "d", "ocr_text": "hello", "subjects": ["s"],
        "actions": "a", "meaning": "m", "claims": [], "intent": "i", "humor": "none",
        "entities": {"people": [], "organizations": [], "places": [], "events": [], "platforms": []},
        "tags": ["t"], "date_hints": [], "language": "en", "sensitivity": 0.1,
    }


def client_for(responses, seen):
    it = iter(responses)

    def handler(request):
        seen.append(json.loads(request.content))
        return next(it)

    return httpx.Client(transport=httpx.MockTransport(handler))


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                     "usage": {"completion_tokens": 42}})


def test_prepare_image_downscales_without_upscaling():
    big = io.BytesIO()
    Image.new("RGBA", (3000, 1000)).save(big, "PNG")
    with Image.open(io.BytesIO(vlm.prepare_image(big.getvalue(), 1024))) as im:
        assert (im.format, im.size) == ("JPEG", (1024, 341))
    small = io.BytesIO()
    Image.new("RGB", (200, 100)).save(small, "PNG")
    with Image.open(io.BytesIO(vlm.prepare_image(small.getvalue(), 1024))) as im:
        assert im.size == (200, 100)


def test_extract_parses_and_sends_schema(monkeypatch):
    monkeypatch.setattr(vlm, "RETRY_SLEEP", 0)
    seen = []
    client = client_for([httpx.Response(503, text="loading"), ok(json.dumps(sample_result()))], seen)
    result = vlm.extract(client, CFG, b"jpg", vlm.IMAGE_SCHEMA, vlm.IMAGE_SYSTEM_PROMPT, "Catalogue this image.")
    assert result["ocr_text"] == "hello"
    assert result["_usage"] == {"completion_tokens": 42}
    assert len(seen) == 2
    body = seen[-1]
    assert body["model"] == "m" and body["max_tokens"] == 100
    assert body["response_format"]["json_schema"]["schema"] is not None
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_extract_raises_on_bad_json(monkeypatch):
    client = client_for([ok("{not json")], [])
    with pytest.raises(vlm.VLMError):
        vlm.extract(client, CFG, b"jpg", vlm.IMAGE_SCHEMA, "sys", "u")


def test_extract_raises_on_4xx():
    client = client_for([httpx.Response(400, text="bad request")], [])
    with pytest.raises(vlm.VLMError, match="bad request"):
        vlm.extract(client, CFG, b"jpg", vlm.IMAGE_SCHEMA, "sys", "u")
