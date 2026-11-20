#!/usr/bin/env python3
"""Metrics for the exported Isambard CTC chunk bitstreams (VTM / ECM / AV1 / AVM; resumable).

Inputs (codec_results/ as unpacked from the *_bitstreams.tar.gz + decoding_toolkit.tar.gz):
  manifest.csv           one row per chunk: codec, preset, class, sequence, rate, first/last frame, size, bitdepth, bitstream
  sequence_coverage.csv  full_sequence=True when the chunks cover the whole configured sequence
  dataset_config.json    class -> base_path of the source YUV
  decoders/<codec>/decoder + decoders/lib
Source YUVs: <source>/<base_path>/<sequence>.yuv (8-bit yuv420p or 10-bit yuv420p10le, as in the manifest).

For each (codec, preset, class, sequence, rate) with full coverage, all chunks are decoded in frame order into one
temporary YUV and compared with the source at native bit depth:
  bpp                       total bitstream bytes * 8 / (W * H * frames)
  psnr_y/u/v, psnr_yuv      (6*Y+U+V)/8, peak 2^bitdepth-1
  msssim_y, msssim_yuv      (6*Y+U+V)/8 (msssim_yuv NaN if a chroma plane is < 161 px)
  vmaf, vmaf_neg            libvmaf vmaf_v0.6.1 / vmaf_v0.6.1neg on the YUV
  lpips_alex                on RGB24 converted from both YUVs (--matrix, limited range)
For 1920x1080 sequences the decoder wall time (dec_fps) and peak RSS are recorded via /usr/bin/time.
Results are cached per sequence/rate as <out>/<codec>/<preset>/<class>/<sequence>/<rate>.json; the decoded YUV is deleted.
"""
import argparse, csv, json, math, os, shutil, subprocess, tempfile
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from collections import defaultdict
from pathlib import Path
import numpy as np

HOME = Path.home()
METRICS = ['psnr_y', 'psnr_u', 'psnr_v', 'psnr_yuv', 'msssim_y', 'msssim_yuv', 'vmaf', 'vmaf_neg', 'lpips_alex']
DATASET = {'UVG': 'UVG', 'MCL-JCV': 'MCL-JCV'}  # HEVC_B..E -> HEVC-CTC, AOM-CTC-A2..A5 -> AOM-CTC


def dataset_of(cls):
    return DATASET.get(cls, 'HEVC-CTC' if cls.startswith('HEVC') else 'AOM-CTC')


def decode(a, rows, recon, tmp):
    """Decode chunks (independent bitstreams) with --dec-jobs parallel decoders, then concatenate them by first_frame.
    Returns summed per-chunk decoder wall seconds and peak RSS (KB)."""
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, LD_LIBRARY_PATH=f"{a.toolkit/'decoders/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")

    def one(i, r):
        depth, part = int(r['bitdepth']), tmp/f'chunk{i:05}.yuv'
        dec, bs = a.toolkit/'decoders'/r['codec']/'decoder', a.toolkit/r['bitstream']
        cmd = ([dec, '--rawvideo', '--i420', f'--output-bit-depth={depth}', '-o', part, bs] if r['codec'] in ('av1', 'avm')
               else [dec, '-b', bs, '-o', part, f'--OutputBitDepth={depth}', f'--OutputBitDepthC={depth}'])
        with open(tmp/f'decode{i:05}.log', 'w') as log:
            subprocess.run(['/usr/bin/time', '-f', '%e %M', '-o', tmp/f'time{i:05}', *map(str, cmd)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        expected = int(r['width'])*int(r['height'])*3//2*(2 if depth > 8 else 1)*int(r['frames'])
        assert part.stat().st_size == expected, f"size mismatch decoding {r['bitstream']}"
        e, m = (tmp/f'time{i:05}').read_text().split()[-2:]
        (tmp/f'decode{i:05}.log').unlink(); (tmp/f'time{i:05}').unlink()
        return float(e), int(m)

    with ThreadPoolExecutor(a.dec_jobs) as ex:
        stats = list(ex.map(one, range(len(rows)), rows))
    with open(recon, 'wb') as out:
        for i in range(len(rows)):
            with open(tmp/f'chunk{i:05}.yuv', 'rb') as f:
                shutil.copyfileobj(f, out)
            (tmp/f'chunk{i:05}.yuv').unlink()
    return sum(e for e, _ in stats), max(m for _, m in stats)


def rgb_frames(a, path, fmt, w, h, n, skip=0):
    """Stream n RGB24 frames from a raw YUV file, starting at frame `skip`."""
    fsize = w*h*3//2*(2 if fmt.endswith('10le') else 1)
    cmd = [a.ffmpeg, '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', fmt, '-s', f'{w}x{h}',
           '-skip_initial_bytes', str(skip*fsize), '-i', str(path), '-frames:v', str(n),
           '-vf', f'scale=in_range=limited:out_range=full:in_color_matrix={a.matrix}:flags=bicubic',
           '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for _ in range(n):
        buf = p.stdout.read(w*h*3)
        assert len(buf) == w*h*3, f'short read from {path}'
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.stdout.close(); p.wait()


def metrics(a, src, recon, w, h, depth, first, n, models, tmp):
    import torch
    from pytorch_msssim import ms_ssim
    alex = models
    fmt, dt, peak = ('yuv420p10le', np.uint16, 1023) if depth > 8 else ('yuv420p', np.uint8, 255)
    fpix = w*h*3//2
    R = np.memmap(src, dt, 'r', offset=first*fpix*dt().itemsize, shape=(n*fpix,))
    D = np.memmap(recon, dt, 'r', shape=(n*fpix,))

    # VMAF / VMAF NEG in one pass (source trimmed to the same frame range).
    model = 'version=vmaf_v0.6.1\\:name=vmaf|version=vmaf_v0.6.1neg\\:name=vmaf_neg'
    raw = lambda p, skip=0: ['-f', 'rawvideo', '-pix_fmt', fmt, '-s', f'{w}x{h}', '-skip_initial_bytes', str(skip), '-i', str(p)]
    with open(tmp/'vmaf.log', 'w') as log:
        subprocess.run([a.vmaf_ffmpeg, '-nostdin', *raw(recon), *raw(src, first*fpix*dt().itemsize), '-lavfi', f"[0:v]trim=end_frame={n}[d];[1:v]trim=end_frame={n}[r];[d][r]libvmaf=model='{model}':log_fmt=json:log_path={tmp/'vmaf.json'}:n_threads={a.threads}",
                        '-f', 'null', '-'], stdout=log, stderr=subprocess.STDOUT, check=True)
    vm = json.loads((tmp/'vmaf.json').read_text())['frames']
    assert len(vm) == n, f'VMAF returned {len(vm)} frames, expected {n}'

    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)/peak).to(a.device)
    frames = []
    with torch.inference_mode():
        for i, (xr, yr) in enumerate(zip(rgb_frames(a, src, fmt, w, h, n, first), rgb_frames(a, recon, fmt, w, h, n))):
            r, d = R[i*fpix:(i+1)*fpix], D[i*fpix:(i+1)*fpix]
            ps, ms = [], []
            for lo, hi, ph, pw in ((0, w*h, h, w), (w*h, w*h*5//4, h//2, w//2), (w*h*5//4, fpix, h//2, w//2)):
                x, y = r[lo:hi].reshape(ph, pw), d[lo:hi].reshape(ph, pw)
                mse = np.mean((x.astype(np.float64)-y)**2)
                ps.append(10*math.log10(peak**2/mse) if mse else 100.0)
                ms.append(ms_ssim(t(x)[None, None], t(y)[None, None], data_range=1.0).item() if min(ph, pw) > 160 else float('nan'))
            X = torch.from_numpy(xr.copy()).permute(2, 0, 1)[None].float().div(255).to(a.device)*2-1
            Y = torch.from_numpy(yr.copy()).permute(2, 0, 1)[None].float().div(255).to(a.device)*2-1
            frames.append(dict(psnr_y=ps[0], psnr_u=ps[1], psnr_v=ps[2], psnr_yuv=(6*ps[0]+ps[1]+ps[2])/8,
                               msssim_y=ms[0], msssim_yuv=(6*ms[0]+ms[1]+ms[2])/8,
                               vmaf=vm[i]['metrics']['vmaf'], vmaf_neg=vm[i]['metrics']['vmaf_neg'],
                               lpips_alex=alex(X, Y).item()))
    del R, D
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--toolkit', type=Path, default=Path(NVC_ROOT, 'codecs/codec_results'))
    p.add_argument('--source', type=Path, default=Path(NVC_ROOT, 'codecs/source_yuv'))
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics'))
    p.add_argument('--codecs', nargs='+', default=['vtm', 'ecm', 'av1', 'avm'])
    p.add_argument('--presets', nargs='+', default=['ra', 'ld'])
    p.add_argument('--sequence', help='only sequences whose name contains this string')
    p.add_argument('--matrix', default='bt709', help='YUV->RGB matrix for LPIPS')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--dec-jobs', type=int, default=4, help='chunks decoded in parallel')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tmp', type=Path, default=None, help='scratch dir for decoded YUV (default: <out>/_tmp)')
    p.add_argument('--ffmpeg', default=str(HOME/'.local/ffmpeg-nvenc/bin/ffmpeg'))
    p.add_argument('--vmaf-ffmpeg', default=str(HOME/'opt/ffmpeg-vmaf/bin/ffmpeg'))
    a = p.parse_args()
    os.environ['LD_LIBRARY_PATH'] = f"{HOME/'.local/ffmpeg-nvenc/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    cfg = json.loads((a.toolkit/'dataset_config.json').read_text())['test_classes']
    cov = list(csv.DictReader((a.toolkit/'sequence_coverage.csv').open()))
    full = {tuple(r[k] for k in ('codec', 'preset', 'class', 'sequence', 'rate')) for r in cov if r['full_sequence'] == 'True'}
    # encoder configuration(s) of each group's chunks: 'original', or e.g. 'fast_cpu9_threads8_rowmt1,original' (mixed AVM)
    confs = {tuple(r[k] for k in ('codec', 'preset', 'class', 'sequence', 'rate')): r.get('configurations') or 'original' for r in cov}
    groups = defaultdict(list)
    for r in csv.DictReader((a.toolkit/'manifest.csv').open()):
        k = (r['codec'], r['preset'], r['sequence_class'], r['sequence'], r['rate'])
        if k in full and r['codec'] in a.codecs and r['preset'] in a.presets and (not a.sequence or a.sequence in r['sequence']) \
                and (a.toolkit/r['bitstream']).is_file():
            groups[k].append(r)
    cache = lambda k: a.out/k[0]/k[1]/k[2]/k[3]/f'{k[4]}.json'
    todo = sorted(k for k in groups if not cache(k).exists())
    print(f'{len(groups)} full sequence/rate points, {len(todo)} to compute', flush=True)

    if todo:
        import lpips
        models = lpips.LPIPS(net='alex', verbose=False).eval().to(a.device)
        scratch = a.tmp or a.out/'_tmp'; scratch.mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(todo, 1):
            rows = sorted(groups[k], key=lambda r: int(r['first_frame']))
            w, h, depth = int(rows[0]['width']), int(rows[0]['height']), int(rows[0]['bitdepth'])
            first, n = int(rows[0]['first_frame']), sum(int(r['frames']) for r in rows)
            assert int(rows[-1]['last_frame']) - first + 1 == n, f'non-contiguous chunks for {k}'
            src = a.source/cfg[k[2]]['base_path']/f'{k[3]}.yuv'
            tmp = Path(tempfile.mkdtemp(dir=scratch))
            try:
                dec_s, dec_kb = decode(a, rows, tmp/'recon.yuv', tmp)
                frames = metrics(a, src, tmp/'recon.yuv', w, h, depth, first, n, models, tmp)
                nbytes = sum(int(r['bitstream_bytes']) for r in rows)
                res = dict(codec=k[0], preset=k[1], cls=k[2], sequence=k[3], rate=k[4], width=w, height=h, bitdepth=depth,
                           first_frame=first, frames=n, chunks=len(rows), bytes=nbytes, bpp=nbytes*8/(w*h*n),
                           **{m: float(np.nanmean([f[m] for f in frames])) for m in METRICS})
                if (w, h) == (1920, 1080):  # decoder throughput / peak RAM, 1080p only
                    res.update(dec_seconds=dec_s, dec_fps=n/dec_s, dec_peak_ram_mb=dec_kb/1024)
                cache(k).parent.mkdir(parents=True, exist_ok=True)
                cache(k).with_suffix('.part').write_text(json.dumps(res, indent=1)); cache(k).with_suffix('.part').replace(cache(k))
                print(f"[{i}/{len(todo)}] {' '.join(k)}  bpp {res['bpp']:.4f}  PSNR-YUV {res['psnr_yuv']:.2f}  VMAF {res['vmaf']:.1f}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] FAIL {' '.join(k)}: {e}  (logs kept in {tmp})", flush=True)
                (tmp/'recon.yuv').unlink(missing_ok=True)
                continue
            shutil.rmtree(tmp)

    # Aggregate: dataset = mean over sequences per codec / preset / rate.
    rows = [dict(json.loads(cache(k).read_text()), configurations=confs[k]) for k in sorted(groups) if cache(k).exists()]
    if not rows:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'ctc_metrics_sequences.csv').open('w', newline='') as f:
        cols = [c for c in dict.fromkeys(k for r in rows for k in r) if c != 'lpips_vgg']  # early caches also hold lpips_vgg
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[(dataset_of(r['cls']), r['codec'], r['preset'], r['rate'])].append(r)
    with (a.out/'ctc_metrics_datasets.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'codec', 'preset', 'rate', 'sequences', 'bpp', *METRICS, 'dec_fps_1080p', 'dec_peak_ram_mb_1080p'])
        for k, v in sorted(agg.items()):
            hd = [r for r in v if 'dec_fps' in r]  # fps = total frames / total decode time over the 1080p sequences
            w.writerow([*k, len(v), np.mean([r['bpp'] for r in v]), *(np.nanmean([r[m] for r in v]) for m in METRICS),
                        sum(r['frames'] for r in hd)/sum(r['dec_seconds'] for r in hd) if hd else '', max((r['dec_peak_ram_mb'] for r in hd), default='')])
    print(f"wrote {a.out/'ctc_metrics_sequences.csv'} ({len(rows)} rows) and ctc_metrics_datasets.csv", flush=True)


if __name__ == '__main__':
    main()
