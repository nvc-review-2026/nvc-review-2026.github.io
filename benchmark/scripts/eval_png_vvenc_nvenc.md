Run from `benchmark/`:

```bash
mkdir -p logs
python -u scripts/eval_png_vvenc_nvenc.py --chunk 32 --frames 96 --codecs vvenc nvenc 2>&1 | tee -a logs/png_vvenc_nvenc_96.log
```

The input root defaults to `$PNG_ROOT` (default `PNG/BT.601`).
All UVG resolutions, MCL-JCV, HEVC-CTC and AOM-CTC sequences are discovered automatically. JVET is excluded. QPs: 22, 27, 32, 37. First 96 naturally sorted PNGs, divided into three independently encoded/decoded 32-frame chunks. Sequences with fewer than 96 frames are recorded in the manifest and excluded.

Both encoders receive the same RGB-to-limited-range-BT.601 8-bit YUV420 conversion. VVenC uses its native 10-bit internal coding, with decoded YUV converted back to 8-bit YUV420 for comparison. Metrics are FFmpeg PSNR (Y/U/V and pixel-weighted YUV) and SSIM on YUV420. Sequence PSNR combines chunk MSE rather than averaging decibels. Bitstreams, exact command arrays, encoder/decoder logs, per-frame metric logs and results JSONs are retained. Large intermediate raw video is removed after each sequence. Decoder timing excludes VVenC's output depth conversion and metrics. NVENC decoding uses FFmpeg's software HEVC decoder.

LD: VVenC GOPSize=8, PicReordering=0, refresh type none, initial IDR; NVENC no B frames, zero lookahead, zero latency. VVenC LD uses B slices with references to already decoded pictures (no frame reordering). RA: VVenC hierarchical GOPSize=32, IDR refresh, frame reordering; NVENC three B frames and middle B reference mode. NVENC RA is a hardware-supported B-frame configuration; its temporal hierarchy differs from VVenC. VVenC preset medium, QPA disabled, 8 threads; NVENC preset p5, constant QP.

Progress and ETA print every 30 seconds and after each chunk. Estimates use measured seconds per pixel-frame for each codec/mode. Uncalibrated modes use provisional same-codec timings. Startup estimates are provisional until completed chunks exist; encoding logs also contain VVenC's own ETA. Estimates vary with content and QP.

PNG files have no timestamps: UVG uses 120 fps; MCL-JCV uses an explicitly recorded 30 fps assumption; other sequences parse fps from names (including 24 fps and fractional rates). These choices affect nominal rate and reported timing ratios, not bpp or PSNR/SSIM.

Outputs: `results/png_bt601_gop32_96/{manifest.json,blocked.json,results_chunks.csv,results_sequences.csv}`. CSVs update after each completed chunk. Re-running skips completed result JSONs. NVENC is probed before the batch: if GPU access fails, the reason is recorded and VVenC continues. To run the blocked NVENC work from a GPU-enabled WSL terminal:

```bash
python -u scripts/eval_png_vvenc_nvenc.py --chunk 32 --frames 96 --codecs nvenc 2>&1 | tee -a logs/png_nvenc_96.log
```

Use distinct output roots if running two processes simultaneously; the default output root should have only one writer. To resume both and export combined CSVs after the VVenC process finishes, run the first command.
