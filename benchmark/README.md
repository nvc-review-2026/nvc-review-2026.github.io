# Benchmark code

Scripts used for the benchmark in *Advances in Neural Video Compression: A Review and Benchmarking*.
Run them from this directory, e.g. `python scripts/eval_ctc_chunks.py --help`.

## Paths

Local paths are read from environment variables:

| Variable | Default | Contents |
|---|---|---|
| `NVC_ROOT` | `VideoCompression` | source YUV, CTC bitstreams/toolkit, encoders, checkpoints, temp files |
| `PNG_ROOT` | `PNG/BT.601` | PNG source sequences (RGB evaluation) |
| `DCVC_ROOT` | `DCVC` | clone of the DCVC repository |
| `VMAF_FFMPEG` | `ffmpeg` | ffmpeg built with libvmaf |
| `PERCEPTUALLOSS_ROOT` | `PerceptualLoss` | repository used by `coolchic_video_metrics_bits.py` |

## Scripts

- Conventional codecs (CTC): `build_ctc_encoders.sh`, `eval_ctc_chunks.py`, `run_ctc_all.sh`, `bench_ctc_encode.py`, `bench_ctc_decode.py`
- VVenC / NVENC: `eval_ctc_vvenc_nvenc.py`, `eval_png_vvenc_nvenc.py`, `bench_vvenc_nvenc.py`
- Learned codecs: `eval_dcvc.py`, `bench_dcvc_gpu.py` (DCVC family, BRHVC), `bench_nycu_gpu.py` (MaskCRT, B-CANF), `bench_inr_gpu.py` (HiNeRV, NVRC), `coolchic_video_metrics_bits.py` + `coolchic_bpp_bits.patch` (C3 / Cool-chic)
- Metrics and BD-rate: `eval_yuv_metrics.py`, `eval_x265_vtm_rgb.py`, `bdrate_ctc.py`, `bdrate_learned.py`
- Profiling: `peakmon.py` (wall time, peak RAM, peak GPU memory)
- Analysis and figures: `build_rd_csv.py`, `summarize_rd_datasets.py`, `siti_profile.py`, `plot_paper_figures.py`, `plot_rd_three_methods.py`, `plot_siti_paper.py`

No results or data are included.
