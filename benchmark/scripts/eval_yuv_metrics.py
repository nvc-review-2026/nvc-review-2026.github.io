#!/usr/bin/env python3
"""YUV-domain metrics for already-encoded chunk bitstreams (reusable, resumable).

Expected layout (as written by eval_png_vvenc_nvenc.py):
  <root>/<dataset>/<group>/<sequence>/<codec>/<mode>/q<qp>/chunkNN/{stream.266|stream.hevc, result.json}
result.json must give first_frame, frames, width, height. The reference is rebuilt from the
source PNGs with the same RGB -> limited-range YUV420 8-bit conversion used for encoding.

Metrics (per frame, then averaged over all frames of the sequence):
  psnr_y/u/v, psnr_yuv = (6*Y+U+V)/8   on 8-bit YUV420
  msssim_y, msssim_yuv = (6*Y+U+V)/8   on 8-bit YUV420 planes (msssim_yuv NaN if a plane is < 161 px)
  lpips (AlexNet), dists               on RGB converted from each YUV420 (same matrix both sides)
  vmaf, vmaf_neg (vmaf_v0.6.1 / vmaf_v0.6.1neg) via libvmaf on YUV420
Per-chunk results are cached as yuv_metrics.json next to result.json; re-running skips them.
"""
import argparse, csv, json, math, os, re, shutil, subprocess, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np

HOME = Path.home()
ENV = dict(os.environ)
ENV['LD_LIBRARY_PATH'] = ':'.join([str(HOME/'.local/ffmpeg-nvenc/lib'), ENV.get('LD_LIBRARY_PATH', '')])
METRICS = ['psnr_y', 'psnr_u', 'psnr_v', 'psnr_yuv', 'msssim_y', 'msssim_yuv', 'lpips', 'dists', 'vmaf', 'vmaf_neg']


def natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name)]


def run(cmd, log):
    with open(log, 'w') as f:
        subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, check=True, env=ENV)


def raw_in(path, w, h, fmt='yuv420p'):
    return ['-f', 'rawvideo', '-pix_fmt', fmt, '-s', f'{w}x{h}', '-i', path]


def planes(buf, w, h):
    y = buf[:w*h].reshape(h, w)
    u = buf[w*h:w*h*5//4].reshape(h//2, w//2)
    v = buf[w*h*5//4:].reshape(h//2, w//2)
    return y, u, v


def chunk_metrics(a, chunk, info, png_dir, models, tmp):
    import torch
    from pytorch_msssim import ms_ssim
    lp, dists = models
    w, h, n, first = info['width'], info['height'], info['frames'], info['first_frame']
    fsize = w*h*3//2
    ref, dec, rgb_ref, rgb_dec = (tmp/x for x in ('ref.yuv', 'dec.yuv', 'ref.rgb', 'dec.rgb'))

    # Reference: identical conversion to the encoding pipeline.
    images = sorted(png_dir.glob('*.png'), key=natural)[first:first+n]
    assert len(images) == n, f'{png_dir}: need {n} PNGs from {first}'
    for i, p in enumerate(images):
        (tmp/f'{i:06}.png').symlink_to(p)
    run([a.ffmpeg, '-nostdin', '-y', '-v', 'warning', '-i', tmp/'%06d.png', '-frames:v', n,
         '-vf', f'scale=in_range=full:out_range=limited:out_color_matrix={a.matrix}:flags=bicubic,format=yuv420p',
         '-f', 'rawvideo', ref], tmp/'ref.log')

    # Decode to 8-bit YUV420.
    stream = next(chunk.glob('stream.*'))
    if stream.suffix == '.266':
        run([a.vvdec, '-b', stream, '-o', tmp/'dec.y4m', '-t', a.threads], tmp/'vvdec.log')
        run([a.ffmpeg, '-nostdin', '-y', '-v', 'warning', '-i', tmp/'dec.y4m', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', dec], tmp/'norm.log')
        (tmp/'dec.y4m').unlink()
    else:
        run([a.ffmpeg, '-nostdin', '-y', '-v', 'warning', '-i', stream, '-pix_fmt', 'yuv420p', '-f', 'rawvideo', dec], tmp/'dec.log')
    for f in (ref, dec):
        assert f.stat().st_size == fsize*n, f'frame count mismatch: {f}'

    # RGB views of both YUV files for LPIPS/DISTS.
    for src, dst in ((ref, rgb_ref), (dec, rgb_dec)):
        run([a.ffmpeg, '-nostdin', '-y', '-v', 'warning', *raw_in(src, w, h),
             '-vf', f'scale=in_range=limited:out_range=full:in_color_matrix={a.matrix}:flags=bicubic',
             '-pix_fmt', 'rgb24', '-f', 'rawvideo', dst], tmp/'rgb.log')

    # VMAF and VMAF NEG in one pass.
    model = 'version=vmaf_v0.6.1\\:name=vmaf|version=vmaf_v0.6.1neg\\:name=vmaf_neg'
    run([a.vmaf_ffmpeg, '-nostdin', '-v', 'info', *raw_in(dec, w, h), *raw_in(ref, w, h),
         '-lavfi', f"[0:v][1:v]libvmaf=model='{model}':log_fmt=json:log_path={tmp/'vmaf.json'}:n_threads={a.threads}",
         '-f', 'null', '-'], tmp/'vmaf.log')
    vm = json.loads((tmp/'vmaf.json').read_text())['frames']
    assert len(vm) == n

    R = np.memmap(ref, np.uint8, 'r'); D = np.memmap(dec, np.uint8, 'r')
    RR = np.memmap(rgb_ref, np.uint8, 'r', shape=(n, h, w, 3)); DR = np.memmap(rgb_dec, np.uint8, 'r', shape=(n, h, w, 3))
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().div(255).to(a.device)
    frames = []
    with torch.inference_mode():
        for i in range(n):
            pr = planes(R[i*fsize:(i+1)*fsize], w, h); pd = planes(D[i*fsize:(i+1)*fsize], w, h)
            ps, ms = [], []
            for x, y in zip(pr, pd):
                mse = np.mean((x.astype(np.float64)-y)**2)
                ps.append(10*math.log10(255**2/mse) if mse else 100.0)
                # 5-scale MS-SSIM needs min side > 160; small chroma planes (e.g. 416x240, 480x270) get NaN.
                ms.append(ms_ssim(t(x)[None, None], t(y)[None, None], data_range=1.0).item() if min(x.shape) > 160 else float('nan'))
            x = t(RR[i]).permute(2, 0, 1)[None]; y = t(DR[i]).permute(2, 0, 1)[None]
            frames.append(dict(psnr_y=ps[0], psnr_u=ps[1], psnr_v=ps[2], psnr_yuv=(6*ps[0]+ps[1]+ps[2])/8,
                               msssim_y=ms[0], msssim_yuv=(6*ms[0]+ms[1]+ms[2])/8,
                               lpips=lp(x*2-1, y*2-1).item(), dists=dists(x, y).item(),
                               vmaf=vm[i]['metrics']['vmaf'], vmaf_neg=vm[i]['metrics']['vmaf_neg']))
    del R, D, RR, DR
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('results/png_bt601_gop32_96'))
    p.add_argument('--png-root', type=Path, default=Path(os.environ.get('PNG_ROOT', 'PNG/BT.601')))
    p.add_argument('--datasets', nargs='+', default=['UVG', 'MCL-JCV', 'HEVC-CTC', 'AOM-CTC'])
    p.add_argument('--codecs', nargs='+', default=['vvenc', 'nvenc'])
    p.add_argument('--modes', nargs='+', default=['LD', 'RA'])
    p.add_argument('--qps', nargs='+', type=int, default=[22, 27, 32, 37])
    p.add_argument('--sequence', help='only sequences whose name contains this string')
    p.add_argument('--matrix', default='bt601')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tmp', type=Path, default=None, help='scratch dir for decoded YUV (default: under --root)')
    p.add_argument('--ffmpeg', default=str(HOME/'.local/ffmpeg-nvenc/bin/ffmpeg'))
    p.add_argument('--vmaf-ffmpeg', default=str(HOME/'opt/ffmpeg-vmaf/bin/ffmpeg'))
    p.add_argument('--vvdec', default=str(HOME/'.local/bin/vvdecapp'))
    a = p.parse_args()

    chunks = sorted(c for ds in a.datasets for c in (a.root/ds).glob('*/*/*/*/q*/chunk*')
                    if c.parts[-4] in a.codecs and c.parts[-3] in a.modes and int(c.parts[-2][1:]) in a.qps
                    and (not a.sequence or a.sequence in c.parts[-5]) and (c/'result.json').exists())
    todo = [c for c in chunks if not (c/'yuv_metrics.json').exists()]
    print(f'{len(chunks)} chunks, {len(todo)} to compute', flush=True)

    if todo:
        import torch, lpips
        from DISTS_pytorch import DISTS
        models = (lpips.LPIPS(net='alex', verbose=False).eval().to(a.device), DISTS().eval().to(a.device))
        scratch = a.tmp or a.root/'_yuv_metrics_tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        for k, c in enumerate(todo, 1):
            rel = c.relative_to(a.root)
            ds, group, seq = rel.parts[0], rel.parts[1], rel.parts[2]
            info = json.loads((c/'result.json').read_text())
            tmp = Path(tempfile.mkdtemp(dir=scratch))
            try:
                frames = chunk_metrics(a, c, info, a.png_root/ds/group/seq, models, tmp)
                out = dict(first_frame=info['first_frame'], frames=info['frames'], width=info['width'],
                           height=info['height'], bytes=info['bytes'], per_frame=frames)
                (c/'yuv_metrics.json.part').write_text(json.dumps(out))
                (c/'yuv_metrics.json.part').replace(c/'yuv_metrics.json')
                print(f'[{k}/{len(todo)}] {rel}  PSNR-YUV {np.mean([f["psnr_yuv"] for f in frames]):.2f}  '
                      f'VMAF {np.mean([f["vmaf"] for f in frames]):.1f}', flush=True)
            except Exception as e:
                print(f'[{k}/{len(todo)}] FAIL {rel}: {e}  (logs kept in {tmp})', flush=True)
                continue
            shutil.rmtree(tmp)

    # Aggregate: sequence = all frames of all chunks; dataset = mean over sequences.
    seqs = defaultdict(lambda: dict(frames=[], bytes=0, pixels=0, nchunks=0))
    for c in chunks:
        f = c/'yuv_metrics.json'
        if not f.exists():
            continue
        m = json.loads(f.read_text()); rel = c.relative_to(a.root).parts
        s = seqs[(rel[0], rel[1], rel[2], rel[3], rel[4], int(rel[5][1:]))]
        s['frames'] += m['per_frame']; s['bytes'] += m['bytes']; s['pixels'] += m['width']*m['height']*m['frames']; s['nchunks'] += 1
    rows = []
    for (ds, group, seq, codec, mode, qp), s in sorted(seqs.items()):
        r = dict(dataset=ds, group=group, sequence=seq, codec=codec, mode=mode, qp=qp,
                 chunks=s['nchunks'], frames=len(s['frames']), bytes=s['bytes'], bpp=s['bytes']*8/s['pixels'])
        r.update({k: float(np.nanmean([f[k] for f in s['frames']])) for k in METRICS})
        rows.append(r)
    if not rows:
        return
    with (a.root/'yuv_metrics_sequences.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['codec'], r['mode'], r['qp'])].append(r)
    with (a.root/'yuv_metrics_datasets.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'codec', 'mode', 'qp', 'sequences', 'bpp', *METRICS])
        for k, v in sorted(groups.items()):
            w.writerow([*k, len(v), np.mean([r['bpp'] for r in v]), *(np.nanmean([r[m] for r in v]) for m in METRICS)])
    print(f'wrote {a.root/"yuv_metrics_sequences.csv"} ({len(rows)} rows) and yuv_metrics_datasets.csv', flush=True)


if __name__ == '__main__':
    main()
