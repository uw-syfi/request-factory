from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from benchmarks import vbench


def _requests(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_t2i_materialization_is_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "subject_consistency.txt").write_text(
        "red fox\nblue whale\ngreen bird\n"
    )
    first = vbench.materialize_vbench(
        vbench.VBenchConfig("t2i", dataset, tmp_path / "first", limit=2, seed=9)
    )
    second = vbench.materialize_vbench(
        vbench.VBenchConfig("t2i", dataset, tmp_path / "second", limit=2, seed=9)
    )

    assert first.selected_examples == 2
    assert first.requests_path.read_bytes() == second.requests_path.read_bytes()
    requests = _requests(first.requests_path)
    assert all(request["inputs"][0]["type"] == "text" for request in requests)
    assert requests[0]["outputs"] == [
        {"type": "image", "width": 1024, "height": 1024, "steps": 50, "count": 1}
    ]
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["selection"]["selected_examples"] == 2
    assert manifest["request"]["client_media_transform"] == "none"


def test_i2i_uses_original_assets_and_scales_output_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "vbench2_beta_i2v"
    origin = dataset / "data" / "origin"
    origin.mkdir(parents=True)
    rows = []
    for index, (width, height) in enumerate(((4000, 2000), (600, 900), (1024, 1024))):
        name = f"image-{index}.jpg"
        (origin / name).write_bytes(b"\xff\xd8" + bytes([index]) * 31 + b"\xff\xd9")
        rows.append(
            {
                "file_name": name,
                "caption": f"caption {index}",
                "origin_width": width,
                "origin_height": height,
            }
        )
    (dataset / "data" / "i2v-bench-info.json").write_text(json.dumps(rows))

    result = vbench.materialize_vbench(
        vbench.VBenchConfig("i2i", dataset, tmp_path / "artifact", seed=0)
    )
    requests = _requests(result.requests_path)
    by_id = {request["id"]: request for request in requests}
    assert by_id["vbench-i2i-0000"]["outputs"][0]["width"] == 1024
    assert by_id["vbench-i2i-0000"]["outputs"][0]["height"] == 512
    assert by_id["vbench-i2i-0001"]["outputs"][0]["width"] == 683
    assert by_id["vbench-i2i-0001"]["outputs"][0]["height"] == 1024
    assert by_id["vbench-i2i-0001"]["outputs"][0]["cfg_img_scale"] == 2.0
    assert by_id["vbench-i2i-0001"]["outputs"][0]["cfg_renorm_type"] == "text_channel"
    assert by_id["vbench-i2i-0001"]["outputs"][0]["cfg_interval"] == [0.0, 1.0]
    for request in requests:
        asset = request["inputs"][0]["asset"]
        source = (result.requests_path.parent / asset["path"]).resolve()
        assert source.parent == origin.resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == asset["sha256"]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["request"]["output_size"] == {
        "long_edge": 1024,
        "preserve_aspect_ratio": True,
    }
    assert manifest["request"]["i2i_cfg"] == {
        "cfg_img_scale": 2.0,
        "cfg_renorm_type": "text_channel",
        "cfg_interval": [0.0, 1.0],
    }
    assert "VBench-I2V" in manifest["provenance_note"]


def test_download_t2i_checks_official_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(b"unexpected prompts\n")

    monkeypatch.setattr(vbench, "_download_url", fake_download)
    with pytest.raises(ValueError, match="unexpected VBench prompt hash"):
        vbench.download_vbench("t2i", tmp_path)


def test_safe_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("../escape.jpg", b"bad")
    with pytest.raises(ValueError, match="escapes destination"):
        vbench._safe_extract_zip(archive, tmp_path / "out")


def test_i2i_fails_closed_when_an_asset_is_missing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "i2v-bench-info.json").write_text(
        json.dumps(
            [
                {
                    "file_name": "missing.jpg",
                    "caption": "something",
                    "origin_width": 10,
                    "origin_height": 20,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="source image is missing"):
        vbench.materialize_vbench(
            vbench.VBenchConfig("i2i", tmp_path, tmp_path / "artifact")
        )
