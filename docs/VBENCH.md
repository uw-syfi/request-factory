# VBench image-generation replay

This adapter materializes the VBench inputs reused by M* and vLLM-Omni for
BAGEL serving experiments. It creates load-generator inputs, not VBench quality
scores:

- `t2i` uses the official VBench `subject_consistency` text prompts;
- `i2i` uses original images and captions from the official VBench-I2V image
  suite.

That second source deserves care: VBench-I2V provides image/caption pairs for
image-to-video evaluation. The public M* harness uses those captions directly
as BAGEL I2I prompts, but they are not a purpose-built editing-instruction set.
The generated manifest records this distinction; this reproduces M*'s serving
inputs without turning them into a claim about image-editing quality.

## Materialize

The interface is:

```bash
# VBench prompts -> 1024x1024 images, 50 steps
uv run python -m benchmarks vbench \
  --task t2i \
  --dataset-dir data/vbench \
  --output-dir out/vbench-t2i \
  --download --limit 160 --seed 0 \
  --arrival-rate 10 --width 1024 --height 1024 --steps 50

# VBench-I2V original image + caption -> image
uv run python -m benchmarks vbench \
  --task i2i \
  --dataset-dir data/vbench \
  --output-dir out/vbench-i2i \
  --download --limit 160 --seed 0 \
  --arrival-rate 10 --i2i-long-edge 1024 --steps 50 \
  --i2i-cfg-img-scale 2 --i2i-cfg-renorm-type text_channel \
  --i2i-cfg-interval 0 1
```

The I2I download follows VBench's official Google Drive release and uses
`gdown`; an installed `gdown` executable is preferred, with `uvx` as the
dependency-isolated fallback. T2I downloads directly from the official GitHub
repository. Known prompt and metadata hashes are checked before use.

You can omit `--download` and point `--dataset-dir` at existing data. Accepted
I2I layouts include either `vbench2_beta_i2v/data/{i2v-bench-info.json,origin/}`
or `data/{i2v-bench-info.json,origin/}`.

## Artifact contract

`requests.jsonl` uses `multimodal-independent-v1`. T2I rows contain text input
and an image output specification. I2I rows contain the original image asset,
its MIME type and SHA-256, followed by the caption. Images are referenced in
place: the materializer does not copy, crop, resize, transcode, or generate
them.

T2I outputs default to the M* settings of 1024×1024 and 50 steps. For I2I, the
requested output dimensions preserve the source metadata's aspect ratio and
scale its long edge to 1024; the system under test remains responsible for
actual preprocessing and generation. The I2I defaults also reproduce the
paper's `cfg_img_scale=2` and `cfg_renorm_type=text_channel` controls. T2I does
not invent a CFG override and leaves that setting to the served model. The
default `cfg_interval=[0,1]` matches the current M* BAGEL adapter (this interval
is an implementation detail not stated in the paper).

`manifest.json` records source metadata hashes, seeded selection, selected
source-index hash, load/generation controls, request hash, selected-asset hash,
and the no-client-transform guarantee.

## Reproducibility boundary

A seed deterministically shuffles source rows before `--limit`. Reusing an
artifact avoids selection drift. Source assets are verified again by the Rust
asset loader before replay.

The adapter intentionally does not implement quality scoring or invent missing
editing instructions. It follows the public M* dataset loader: the 72
`subject_consistency` prompts for T2I and VBench-I2V `origin` images plus
captions for I2I. The paper says I2I preserves aspect ratio with long edge 1024,
while the current public M* request adapter forces 1024×1024; this materializer
follows the paper and records that choice in its manifest.
