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


def test_tile_boxes_cover_tall_image_with_overlap():
    for width, height in [(768, 4070), (400, 2130), (768, 2000), (300, 20000)]:
        boxes = vlm.tile_boxes(width, height, max_tiles=6)
        assert 2 <= len(boxes) <= 6
        assert boxes[0][1] == 0 and boxes[-1][3] == height
        for (_, _, _, prev_bottom), (_, top, _, _) in zip(boxes, boxes[1:]):
            assert top < prev_bottom  # consecutive slices overlap
    assert len(vlm.tile_boxes(768, 4070, 6)) == 5


def test_prepare_images_slices_only_tall_images():
    def png(w, h):
        buf = io.BytesIO()
        Image.new("RGB", (w, h)).save(buf, "PNG")
        return buf.getvalue()

    assert len(vlm.prepare_images(png(1000, 1500))) == 1
    tiles = vlm.prepare_images(png(1170, 6000), max_tiles=6)
    assert len(tiles) > 1
    with Image.open(io.BytesIO(tiles[0])) as im:
        assert im.width == vlm.TILE_WIDTH
    assert "3 overlapping" in vlm.tiles_note(3) and vlm.tiles_note(1) == ""


def test_extract_sends_every_tile_in_order():
    seen = []
    client = client_for([ok(json.dumps(sample_result()))], seen)
    vlm.extract(client, CFG, [b"one", b"two", b"three"], vlm.IMAGE_SCHEMA, "sys", "u")
    parts = seen[0]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url", "image_url"]
    assert parts[3]["image_url"]["url"].endswith("dGhyZWU=")  # b"three"


def test_dense_portrait_images_slice_only_when_asked():
    buf = io.BytesIO()
    Image.new("RGB", (1559, 2318)).save(buf, "PNG")
    assert len(vlm.prepare_images(buf.getvalue())) == 1
    assert len(vlm.prepare_images(buf.getvalue(), dense=True)) >= 2
    square = io.BytesIO()
    Image.new("RGB", (1024, 995)).save(square, "PNG")
    assert len(vlm.prepare_images(square.getvalue(), dense=True)) == 1


def test_length_error_keeps_raw_output():
    client = client_for([httpx.Response(200, json={"choices": [
        {"message": {"content": '{"description": "cut'}, "finish_reason": "length"}]})], [])
    with pytest.raises(vlm.VLMLengthError) as info:
        vlm.extract(client, CFG, b"jpg", vlm.IMAGE_SCHEMA, "sys", "u")
    assert info.value.raw == '{"description": "cut'
