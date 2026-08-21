"""Materialize VBench prompts and image-suite rows for BAGEL replay.

VBench is a video-generation benchmark.  M* and vLLM-Omni reuse its text
prompts for text-to-image serving and the VBench-I2V image suite for
image-conditioned generation.  This module preserves that distinction in the
manifest instead of claiming the I2V captions are image-editing instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

VBenchTask = Literal["t2i", "i2i"]

T2I_PROMPT_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "prompts/prompts_per_dimension/subject_consistency.txt"
)
T2I_PROMPT_SHA256 = "b6b72ab4799acd7fa9bee99c34b938df0c49ca303a0abf38f17ebe15c8085c49"
I2V_INFO_GOOGLE_DRIVE_ID = "1zmWs_m_A4q6YgTZwIZ230jW0ttknlGJA"
I2V_ORIGIN_GOOGLE_DRIVE_ID = "1qhkLCSBkzll0dkKpwlDTwLL0nxdQ4nrY"
I2V_INFO_SHA256 = "7dbe47684eadb2b4ace3fe452a4d8e8ceffa9c788d9c1ff048af3bb2585f96bf"
VBench_REPOSITORY = "https://github.com/Vchitect/VBench"


@dataclass(frozen=True)
class VBenchConfig:
    """Inputs to deterministic VBench request materialization."""

    task: VBenchTask
    dataset_dir: Path
    output_dir: Path
    download: bool = False
    limit: int | None = None
    seed: int = 0
    arrival_rate: float = 1.0
    width: int = 1024
    height: int = 1024
    steps: int = 50
    count: int = 1
    i2i_long_edge: int = 1024
    i2i_cfg_img_scale: float = 2.0
    i2i_cfg_renorm_type: str = "text_channel"
    i2i_cfg_interval: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class MaterializationResult:
    requests_path: Path
    manifest_path: Path
    selected_examples: int


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``vbench`` CLI; intended for benchmarks.__main__."""

    parser = subparsers.add_parser(
        "vbench", help="materialize VBench BAGEL text/image generation requests"
    )
    parser.add_argument("--task", choices=("t2i", "i2i"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--i2i-long-edge", type=int, default=1024)
    parser.add_argument("--i2i-cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--i2i-cfg-renorm-type", default="text_channel")
    parser.add_argument(
        "--i2i-cfg-interval",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=(0.0, 1.0),
    )


def config_from_arguments(arguments: argparse.Namespace) -> VBenchConfig:
    """Translate parsed CLI arguments into the library configuration."""

    return VBenchConfig(
        task=arguments.task,
        dataset_dir=arguments.dataset_dir,
        output_dir=arguments.output_dir,
        download=arguments.download,
        limit=arguments.limit,
        seed=arguments.seed,
        arrival_rate=arguments.arrival_rate,
        width=arguments.width,
        height=arguments.height,
        steps=arguments.steps,
        count=arguments.count,
        i2i_long_edge=arguments.i2i_long_edge,
        i2i_cfg_img_scale=arguments.i2i_cfg_img_scale,
        i2i_cfg_renorm_type=arguments.i2i_cfg_renorm_type,
        i2i_cfg_interval=tuple(arguments.i2i_cfg_interval),
    )


def materialize_from_arguments(arguments: argparse.Namespace) -> int:
    materialize_vbench(config_from_arguments(arguments))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as writer:
        shutil.copyfileobj(response, writer, length=1024 * 1024)
    partial.replace(destination)


def _download_google_drive(file_id: str, destination: Path) -> None:
    """Use gdown's maintained Google Drive confirmation/cookie handling."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if shutil.which("gdown"):
        command = ["gdown", file_id, "--output", str(partial), "--continue"]
    elif shutil.which("uvx"):
        command = [
            "uvx",
            "--from",
            "gdown",
            "gdown",
            file_id,
            "--output",
            str(partial),
            "--continue",
        ]
    else:
        raise ValueError(
            "VBench-I2V download requires gdown; install it or make uvx available"
        )
    subprocess.run(command, check=True)
    if not partial.is_file():
        raise ValueError(f"gdown did not produce {partial}")
    partial.replace(destination)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as reader:
        for member in reader.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(
                    f"archive member escapes destination: {member.filename!r}"
                )
            mode = member.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError(
                    f"archive contains a symbolic link: {member.filename!r}"
                )
        reader.extractall(destination)


def _find_i2v_info(dataset_dir: Path) -> tuple[Path, Path]:
    candidates = (
        dataset_dir / "data" / "i2v-bench-info.json",
        dataset_dir / "vbench2_beta_i2v" / "data" / "i2v-bench-info.json",
        dataset_dir / "i2v-bench-info.json",
    )
    for info in candidates:
        if info.is_file():
            data_dir = info.parent
            return info, data_dir / "origin"
    raise ValueError(
        f"VBench-I2V metadata not found below {dataset_dir}; pass --download or "
        "point --dataset-dir at vbench2_beta_i2v"
    )


def download_vbench(task: VBenchTask, dataset_dir: Path) -> dict[str, str]:
    """Acquire the official inputs needed by a serving workload."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if task == "t2i":
        prompts = dataset_dir / "subject_consistency.txt"
        if not prompts.is_file():
            _download_url(T2I_PROMPT_URL, prompts)
        digest = _sha256(prompts)
        if digest != T2I_PROMPT_SHA256:
            raise ValueError(
                f"unexpected VBench prompt hash: expected {T2I_PROMPT_SHA256}, got {digest}"
            )
        return {"prompt_sha256": digest}

    data_dir = dataset_dir / "vbench2_beta_i2v" / "data"
    info = data_dir / "i2v-bench-info.json"
    if not info.is_file():
        _download_google_drive(I2V_INFO_GOOGLE_DRIVE_ID, info)
    info_digest = _sha256(info)
    if info_digest != I2V_INFO_SHA256:
        raise ValueError(
            f"unexpected VBench-I2V metadata hash: expected {I2V_INFO_SHA256}, got {info_digest}"
        )
    origin = data_dir / "origin"
    archive = data_dir / "origin.zip"
    if not origin.is_dir():
        if not archive.is_file():
            _download_google_drive(I2V_ORIGIN_GOOGLE_DRIVE_ID, archive)
        _safe_extract_zip(archive, data_dir)
    if not origin.is_dir():
        raise ValueError(f"VBench-I2V origin archive did not create {origin}")
    provenance = {"metadata_sha256": info_digest}
    if archive.is_file():
        provenance["origin_archive_sha256"] = _sha256(archive)
    return provenance


def _validate(config: VBenchConfig) -> None:
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit must be greater than zero")
    if config.arrival_rate <= 0:
        raise ValueError("arrival_rate must be greater than zero")
    for name, value in (
        ("width", config.width),
        ("height", config.height),
        ("steps", config.steps),
        ("count", config.count),
        ("i2i_long_edge", config.i2i_long_edge),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if config.i2i_cfg_img_scale <= 0:
        raise ValueError("i2i_cfg_img_scale must be greater than zero")
    if not config.i2i_cfg_renorm_type.strip():
        raise ValueError("i2i_cfg_renorm_type must not be empty")
    interval_start, interval_end = config.i2i_cfg_interval
    if not (
        math.isfinite(interval_start)
        and math.isfinite(interval_end)
        and 0.0 <= interval_start <= interval_end <= 1.0
    ):
        raise ValueError("i2i_cfg_interval must satisfy 0 <= START <= END <= 1")


def _load_t2i(dataset_dir: Path) -> tuple[list[dict[str, object]], Path]:
    prompt_path = dataset_dir / "subject_consistency.txt"
    if not prompt_path.is_file():
        raise ValueError(
            f"VBench prompt file not found at {prompt_path}; pass --download or provide it"
        )
    prompts = [
        line.strip() for line in prompt_path.read_text().splitlines() if line.strip()
    ]
    if not prompts:
        raise ValueError(f"VBench prompt file is empty: {prompt_path}")
    return [
        {"source_index": index, "prompt": prompt}
        for index, prompt in enumerate(prompts)
    ], prompt_path


def _load_i2i(dataset_dir: Path) -> tuple[list[dict[str, object]], Path]:
    info_path, origin_dir = _find_i2v_info(dataset_dir)
    raw = json.loads(info_path.read_text())
    if not isinstance(raw, list):
        # This is malformed user-supplied data, surfaced with the other input errors.
        raise ValueError(  # noqa: TRY004
            f"VBench-I2V metadata must be a JSON list: {info_path}"
        )
    rows: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(  # noqa: TRY004
                f"VBench-I2V row {index} must be an object"
            )
        filename = item.get("file_name")
        caption = item.get("caption")
        width = item.get("origin_width")
        height = item.get("origin_height")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"VBench-I2V row {index} has no file_name")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"VBench-I2V row {index} has no caption")
        if (
            not isinstance(width, int)
            or width <= 0
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError(f"VBench-I2V row {index} has invalid origin dimensions")
        image = origin_dir / filename
        if not image.is_file():
            raise ValueError(f"VBench-I2V source image is missing: {image}")
        rows.append(
            {
                "source_index": index,
                "prompt": caption.strip(),
                "image": image,
                "origin_width": width,
                "origin_height": height,
            }
        )
    if not rows:
        raise ValueError(f"VBench-I2V metadata has no rows: {info_path}")
    return rows, info_path


def _scaled_dimensions(width: int, height: int, long_edge: int) -> tuple[int, int]:
    scale = long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _media_type(image: Path) -> str:
    media_type, _ = mimetypes.guess_type(image.name)
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"unsupported VBench source image type: {image}")
    return media_type


def materialize_vbench(config: VBenchConfig) -> MaterializationResult:
    """Write canonical requests and a provenance manifest without transforming media."""

    _validate(config)
    dataset_dir = config.dataset_dir.expanduser().resolve()
    download_provenance = (
        download_vbench(config.task, dataset_dir) if config.download else {}
    )
    if config.task == "t2i":
        rows, source_metadata = _load_t2i(dataset_dir)
    else:
        rows, source_metadata = _load_i2i(dataset_dir)

    random.Random(config.seed).shuffle(rows)
    if config.limit is not None:
        rows = rows[: config.limit]

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    interval_ms = 1000.0 / config.arrival_rate
    asset_hashes: list[str] = []
    with requests_path.open("w") as requests:
        for request_index, row in enumerate(rows):
            source_index = int(row["source_index"])
            inputs: list[dict[str, object]] = []
            if config.task == "i2i":
                image = Path(row["image"])
                digest = _sha256(image)
                asset_hashes.append(digest)
                inputs.append(
                    {
                        "type": "image",
                        "asset": {
                            "path": Path(os.path.relpath(image, output_dir)).as_posix(),
                            "sha256": digest,
                            "media_type": _media_type(image),
                        },
                    }
                )
                output_width, output_height = _scaled_dimensions(
                    int(row["origin_width"]),
                    int(row["origin_height"]),
                    config.i2i_long_edge,
                )
            else:
                output_width, output_height = config.width, config.height
            inputs.append({"type": "text", "text": row["prompt"]})
            output = {
                "type": "image",
                "width": output_width,
                "height": output_height,
                "steps": config.steps,
                "count": config.count,
                # BAGEL on M* reads `num_timesteps`, while the dialect's general
                # image profile calls the semantic knob `num_inference_steps`.
                # Keep the model exception explicit and scoped to M*.
                "model_params": {"mstar": {"num_timesteps": config.steps}},
            }
            if config.task == "i2i":
                i2i_params = {
                    "cfg_img_scale": config.i2i_cfg_img_scale,
                    "cfg_renorm_type": config.i2i_cfg_renorm_type,
                    "cfg_interval": list(config.i2i_cfg_interval),
                }
                output["model_params"]["mstar"].update(i2i_params)
                output["model_params"]["vllm-omni"] = i2i_params
            request = {
                "id": f"vbench-{config.task}-{source_index:04d}",
                "arrival_time_ms": request_index * interval_ms,
                "inputs": inputs,
                "outputs": [output],
            }
            requests.write(json.dumps(request, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": 1,
        "benchmark": f"bagel-{config.task}-vbench",
        "dataset": {
            "name": "VBench subject-consistency prompts"
            if config.task == "t2i"
            else "VBench-I2V image suite",
            "repository": VBench_REPOSITORY,
            "source_metadata_sha256": _sha256(source_metadata),
            "download": download_provenance,
        },
        "selection": {
            "seed": config.seed,
            "limit": config.limit,
            "selected_examples": len(rows),
            "selected_source_indices_sha256": hashlib.sha256(
                "\n".join(str(row["source_index"]) for row in rows).encode()
            ).hexdigest(),
        },
        "request": {
            "format": "multimodal-independent-v1",
            "task": config.task,
            "arrival_rate_per_s": config.arrival_rate,
            "steps": config.steps,
            "count": config.count,
            "output_size": {"width": config.width, "height": config.height}
            if config.task == "t2i"
            else {"long_edge": config.i2i_long_edge, "preserve_aspect_ratio": True},
            "i2i_cfg": {
                "cfg_img_scale": config.i2i_cfg_img_scale,
                "cfg_renorm_type": config.i2i_cfg_renorm_type,
                "cfg_interval": list(config.i2i_cfg_interval),
            }
            if config.task == "i2i"
            else None,
            "client_media_transform": "none",
        },
        "artifacts": {
            "requests": requests_path.name,
            "requests_sha256": _sha256(requests_path),
            "selected_assets_sha256": hashlib.sha256(
                "\n".join(asset_hashes).encode()
            ).hexdigest()
            if asset_hashes
            else None,
        },
        "provenance_note": (
            "I2I serving rows reuse VBench-I2V origin images and captions; VBench does not "
            "define these captions as image-editing instructions."
            if config.task == "i2i"
            else "M* and vLLM-Omni reuse VBench video prompts for image generation."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {len(rows)} VBench {config.task} requests to {requests_path}",
        file=sys.stderr,
    )
    return MaterializationResult(requests_path, manifest_path, len(rows))
