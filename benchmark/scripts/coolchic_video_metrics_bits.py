#!/usr/bin/env python3
"""Score a Cool-Chic video reconstruction (raw yuv420p) against the original PNG frames
using exactly the NVRC conventions behind Analysis/results.csv:

  psnr  : RGB, [0,1], 10*log10(1/mse), per-frame mean
  lpips : lpips.LPIPS(net="alex"), normalize=True
  dists : DISTS_pytorch.DISTS
  vmaf  : whole-clip libvmaf on BT.601 luma from RGB (NVRC/loss_utils.compute_vmaf_scores)
  bpp   : bits-per-pixel = 8 * bytes / (T*H*W), taken from rate.txt (entropy-model estimate,
          bits/px) unless --bytes is given (real bitstream size)

The yuv is converted to RGB PNGs with ffmpeg defaults, mirroring how the PNG source was
turned into the yuv420p input, so the 4:2:0 round trip is the same both ways.

  python video_metrics.py --run Outputs/CoolChic/video_mse_l0.0004 --tag coolchic_mse
  python video_metrics.py --yuv Outputs/CoolChic/data/kite-long_*.yuv --tag ceiling_420
"""
import argparse
import os
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(os.environ.get("PERCEPTUALLOSS_ROOT", "PerceptualLoss"))
sys.path.insert(0, str(REPO / "NVRC"))
from loss_utils import compute_vmaf_scores  # noqa: E402

W, H = 1920, 1080


def yuv_to_png(yuv: Path, out_dir: Path, n_frames: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.png"))) == n_frames:
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "yuv420p",
         "-video_size", f"{W}x{H}", "-i", str(yuv), "-start_number", "0", str(out_dir / "%04d.png")],
        check=True,
    )


def load_rgb(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="dir containing dec.yuv (+ rate.txt)")
    g.add_argument("--yuv", help="score an arbitrary yuv (e.g. the source, for the 4:2:0 ceiling)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src_png", default=str(REPO / "Datasets" / "kite-long"))
    ap.add_argument("--bytes", type=int, help="real bitstream size, overrides rate.txt")
    ap.add_argument("--out", help="metrics csv (default <run>/metrics.csv)")
    args = ap.parse_args()

    src = sorted(Path(args.src_png).glob("*.png"))
    T = len(src)
    if args.run:
        run = Path(args.run)
        yuv = run / "dec.yuv"
        out_csv = Path(args.out or run / "metrics.csv")
        rgb_dir = run / "rgb"
    else:
        yuv = Path(args.yuv)
        out_csv = Path(args.out or yuv.with_suffix(".metrics.csv"))
        rgb_dir = yuv.with_suffix("") .parent / f"{yuv.stem}_rgb"
    assert yuv.stat().st_size == T * W * H * 3 // 2, f"{yuv}: not {T} frames of {W}x{H} yuv420p"

    yuv_to_png(yuv, rgb_dir, T)
    dec = sorted(rgb_dir.glob("*.png"))
    assert len(dec) == T

    # rate
    bpp_bits = None
    if args.bytes:
        bpp_bits = 8 * args.bytes / (T * W * H)
    elif args.run and (run / "rate.txt").exists():
        m = re.search(r"rate_bpp\(bits/px\)=([0-9.eE+-]+)", (run / "rate.txt").read_text())
        if m:
            bpp_bits = float(m.group(1))

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    import lpips
    from DISTS_pytorch import DISTS
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    ds = DISTS().to(dev).eval()

    rows, ref_y, dis_y = [], [], []
    for i, (s, d) in enumerate(zip(src, dec)):
        x, y = load_rgb(s).to(dev), load_rgb(d).to(dev)
        with torch.no_grad():
            mse = torch.mean((x - y) ** 2).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-9))
            l = lp(x, y, normalize=True).item()
            di = ds(x, y, batch_average=False).mean().item()
        for buf, t in ((ref_y, x), (dis_y, y)):
            a = (t[0].cpu().numpy() * 255)
            buf.append((0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]).round().astype(np.uint8))
        rows.append(dict(frame=i, psnr=psnr, lpips=l, dists=di))
        if i % 10 == 0:
            print(f"  {i:4d}/{T}  psnr={psnr:.2f} lpips={l:.4f} dists={di:.4f}", flush=True)

    print("  vmaf (whole clip)...", flush=True)
    vm = compute_vmaf_scores(np.stack(ref_y), np.stack(dis_y), H, W)
    for r, v in zip(rows, vm):
        r["vmaf"] = v

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "psnr", "lpips", "dists", "vmaf"])
        w.writeheader(); w.writerows(rows)

    m = {k: float(np.nanmean([r[k] for r in rows])) for k in ("psnr", "lpips", "dists", "vmaf")}
    summ = (f"{args.tag}\tframes={T}\tbpp={bpp_bits if bpp_bits is None else f'{bpp_bits:.5f}'}"
            f"\trate_unit=bits/pixel/frame\tpsnr={m['psnr']:.4f}\tlpips={m['lpips']:.4f}\tdists={m['dists']:.4f}\tvmaf={m['vmaf']:.4f}")
    print(summ)
    out_csv.with_name(out_csv.stem + "_summary.txt").write_text(summ + "\n")


if __name__ == "__main__":
    main()
