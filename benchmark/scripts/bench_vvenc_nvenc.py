#!/usr/bin/env python3
"""Encode + decode throughput / peak RAM (and GPU memory, scripts/peakmon.py) of VVenC and NVENC on one GoP of 1080p (run alone, nothing else
on the machine or the GPU).

Same encoder/decoder settings as eval_png_vvenc_nvenc.py (VVenC medium + vvdec, 8 threads; NVENC p5 constqp + FFmpeg
software HEVC decode). Sequence MCL-JCV videoSRC10 (8-bit 1080p, also in the VTM encode bench), QP 27,
GoP = 32 frames RA / 8 frames LD as in bench_ctc_encode.py. One process at a time, each point --repeat times (median).
Writes <out>/vvenc_nvenc_bench.csv.
"""
import argparse, csv, os, statistics, sys, tempfile
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import peakmon  # noqa: E402

HOME = Path.home()
VE, VD = HOME/'.local/bin/vvencapp', HOME/'.local/bin/vvdecapp'
FF = HOME/'.local/ffmpeg-nvenc/bin/ffmpeg'
SRC = Path(NVC_ROOT, 'codecs/source_yuv/MCL-JCV/1920x1080/videoSRC10_1920x1080_yuv420p.yuv')
W, H, FPS = 1920, 1080, '30'
GOP = {'RA': 32, 'LD': 8}


def timed(cmd, gpu):
    r = peakmon.run(cmd, gpu=gpu, interval=0.01)  # LD decodes 8 frames in ~0.1 s
    return r['seconds'], r['host_mib'], r['gpu_mib']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics'))
    p.add_argument('--qp', type=int, default=27)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--gpu', type=int, default=0, help='GPU whose memory.used is tracked (NVENC encode / GPU decode)')
    a = p.parse_args()
    os.environ['LD_LIBRARY_PATH'] = f"{HOME/'.local/ffmpeg-nvenc/lib'}:/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    tmp = Path(tempfile.mkdtemp(dir=a.out))
    rows = []
    for mode, n in GOP.items():
        raw = tmp/'src.yuv'
        with SRC.open('rb') as f:
            raw.write_bytes(f.read(W*H*3//2*n))
        for codec in ('vvenc', 'nvenc'):
            bs = tmp/('stream.266' if codec == 'vvenc' else 'stream.hevc')
            if codec == 'vvenc':
                extra = f'GOPSize={8 if mode == "LD" else 32}:PicReordering={0 if mode == "LD" else 1}:POC0IDR=1:InternalBitDepth=10'
                enc = [VE, '-i', raw, '-s', f'{W}x{H}', '--fps', FPS, '--format', 'yuv420', '--frames', n, '--preset', 'medium',
                       '--threads', a.threads, '--qpa', '0', '--qp', a.qp, '--intraperiod', n,
                       '--refreshtype', 'none' if mode == 'LD' else 'idr', '--additional', extra, '-o', bs]
                dec = [VD, '-b', bs, '-o', tmp/'dec.yuv', '-t', a.threads]
            else:
                enc = [FF, '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{W}x{H}', '-r', FPS,
                       '-i', raw, '-frames:v', n, '-c:v', 'hevc_nvenc', '-preset', 'p5', '-rc', 'constqp', '-qp', a.qp, '-g', n,
                       '-bf', '0' if mode == 'LD' else '3', '-rc-lookahead', '0', '-forced-idr', '1',
                       '-b_ref_mode', 'disabled' if mode == 'LD' else 'middle', '-zerolatency', '1' if mode == 'LD' else '0', bs]
                dec = [FF, '-nostdin', '-y', '-v', 'error', '-i', bs, '-pix_fmt', 'yuv420p', '-f', 'rawvideo', tmp/'dec.yuv']
            es, em, eg, ds, dm, dg = [], [], [], [], [], []
            for _ in range(a.repeat):
                s, m, g = timed(enc, a.gpu); es.append(s); em.append(m); eg.append(g)
                s, m, g = timed(dec, a.gpu); ds.append(s); dm.append(m); dg.append(g)
                (tmp/'dec.yuv').unlink()
            r = dict(codec=codec, mode=mode, sequence=SRC.stem, qp=a.qp, frames=n, bytes=bs.stat().st_size,
                     enc_seconds=statistics.median(es), enc_fps=n/statistics.median(es), enc_peak_ram_mb=max(em), enc_peak_gpu_mib=max(eg),
                     dec_seconds=statistics.median(ds), dec_fps=n/statistics.median(ds), dec_peak_ram_mb=max(dm), dec_peak_gpu_mib=max(dg))
            rows.append(r); bs.unlink()
            print(f"{codec} {mode}: enc {r['enc_fps']:.2f} fps {r['enc_peak_ram_mb']:.0f} MB | "
                  f"dec {r['dec_fps']:.1f} fps {r['dec_peak_ram_mb']:.0f} MB", flush=True)
        raw.unlink()
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    with (a.out/'vvenc_nvenc_bench.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out/'vvenc_nvenc_bench.csv'}", flush=True)


if __name__ == '__main__':
    main()
