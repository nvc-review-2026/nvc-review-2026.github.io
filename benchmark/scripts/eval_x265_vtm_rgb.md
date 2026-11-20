Run from `benchmark/`:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate compression
python -u scripts/eval_x265_vtm_rgb.py \
  --vtm-root /path/to/VTM \
  --frames 96 --qps 22 27 32 37 --mode RA
```

VTM 22.2 was located at `/path/to/VTM`; its Linux encoder
and decoder execute successfully, and its RA config uses GOP32. The build has
`bin/EncoderAppStatic`, `bin/DecoderAppStatic` and its matching
`cfg/encoder_randomaccess_vtm.cfg`. Alternatively use `--vtm-encoder`,
`--vtm-decoder`, and `--vtm-config` for custom paths. Do not substitute VVenC for VTM.
The default protocol expects the GOP32 random-access config. Use the
configuration distributed with the exact encoder release; do not mix releases.

Defaults: all seven UVG sequences at the available 1920x1080 resolution and all
five HEVC-B sequences in `$PNG_ROOT`, first
96 naturally sorted PNGs, one continuous encode per sequence/QP/codec. UVG fps
is 120; HEVC-B fps is read from sequence names (24, 50 or 60). This produces
96 jobs. `--uvg-resolution 1280x720` selects the other available UVG resolution.
These are the local PNG versions, not original UHD UVG YUV sources.

Useful commands:

```bash
# Inspect the 96-job plan without requiring VTM or loading metric models.
python scripts/eval_x265_vtm_rgb.py --plan-only

# x265 only, or VTM only (with --vtm-root).
python -u scripts/eval_x265_vtm_rgb.py --codecs x265 --out results/x265_rgb_96

# One real-sequence check, with a separate output directory.
python -u scripts/eval_x265_vtm_rgb.py --codecs x265 --datasets HEVC-B \
  --sequence Kimono1_1920x1080_24 --qps 32 --frames 2 --preset ultrafast \
  --out results/x265_rgb_smoke
```

CTC interpretation:

- Default QPs are 22, 27, 32, 37. VTM retains the config's temporal QP offsets
  and coding tools, with internal depth 10. RA intra periods are 32 for 24/30
  fps, 64 for both 50 and 60 fps, and 128 for UVG at 120 fps. The script
  requires GOPSize=32 in the complete RA configuration, including all 32
  Frame entries. An old GOP16 config is rejected; changing only its GOPSize
  would not supply the required reference-picture structure.
- x265 uses `veryslow`, `tune=psnr`, constant base QP, fixed 15 B frames,
  B adaptation/scenecut/AQ/CU-tree disabled, B-pyramid enabled, 40-frame
  lookahead, open GOP, and the same intra
  periods. x265 supports at most 16 consecutive B frames and cannot implement
  the VTM GOP32 hierarchy. The selected 15 B frames form a mini-GOP16
  baseline; it must not be reported as GOP32 CTC. Setting keyint=32 alone
  would change the intra period, not reproduce the VTM reference structure.
- `--mode LD` uses VTM's low-delay config (initial intra only); x265 uses
  P-only low delay. `--mode AI` uses the intra config and every-frame intra.
  Choose a separate `--out` for each protocol.
- PNG input, a 96-frame truncation, UVG extensions, and x265's GOP make this
  a **CTC-derived comparison**, not an official full CTC conformance run.
  No independent 32-frame chunks or padding frames are introduced.

Metric definitions:

- Both encoders receive the same limited-range 10-bit YUV420 converted from
  original 8-bit RGB PNGs with bicubic chroma sampling. BT.601 is explicit to
  match this dataset; use `--matrix bt709` only with a corresponding source.
- Decoded 10-bit YUV is converted to full-range RGB24 with the same matrix.
  PSNR, LPIPS and DISTS compare this to **original PNG RGB**, including the
  color-conversion/chroma-subsampling loss. No resize or crop is applied.
- `psnr_rgb` pools MSE over every pixel, RGB channel and frame, peak=255;
  `psnr_rgb_frame_mean` separately averages per-frame dB scores.
- LPIPS uses pretrained AlexNet, input [-1,1]; DISTS uses pretrained VGG16,
  input [0,1]. Both average frame scores. Full-resolution inference may need
  substantial GPU memory; use `--device cpu` if necessary. No automatic
  downscaling changes the metric. Model loading is checked before encoding.
- Standard VMAF is **not an RGB metric**. `vmaf_rgb_luma` uses model
  `vmaf_v0.6.1`, converting the exact RGB pair to limited-range YUV444 for
  libvmaf. It measures derived luma and retains temporal frame order. It is
  neither three-channel RGB VMAF nor VMAF on the codec's original YUV pair.
- `bitrate_kbps = bitstream_bytes * 8 * fps / frames / 1000`, including all
  elementary-stream headers. `bpp` is also reported; no container overhead.

Outputs are in `results/x265_vtm_rgb_96`: `results.csv`, the manifest, exact
commands in logs, actual `.hevc`/`.vvc` bitstreams, per-job `result.json`,
`frames.csv`, and VMAF JSON. Large raw intermediates are cleaned up after each
sequence. Failed jobs stop the run; completed jobs are resumed on rerun.
Changing settings requires a new output directory. Use one writer per output
directory. Keep source PNGs and executable builds unchanged when resuming.
The CSV is refreshed as jobs are visited; it is complete when the run ends.

The script uses `/usr/bin/ffmpeg` for x265 and the installed
`$VMAF_FFMPEG` for VMAF; both are overridable.
Metric Python dependencies are available in `compression` (`torch`, `lpips`,
`DISTS_pytorch`, `numpy`, `Pillow`), including cached model weights.

References: [VTM source/configurations](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM),
[x265 CLI](https://x265.readthedocs.io/en/master/cli.html),
[Netflix VMAF FFmpeg usage](https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md),
[DISTS implementation](https://github.com/dingkeyan93/DISTS).

Validation: dataset discovery found 12 sequences / 96 default jobs. A two-frame
128x128 synthetic x265 encode/decode passed all four metric computations in
`compression`. The VTM 22.2 encoder/decoder startup and GOP32 config have been verified;
a full VTM benchmark has not been run.
