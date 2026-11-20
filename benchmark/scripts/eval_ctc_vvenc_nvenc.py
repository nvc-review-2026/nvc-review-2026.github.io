#!/usr/bin/env python3
"""VVenC / NVENC on the same CTC YUV sequences and the same chunking as the Isambard VTM bitstreams (resumable),
so their BD-rate vs VTM can be computed with bdrate_ctc.py.

For each (preset, class, sequence) that VTM has in <ctc>/ctc_metrics_sequences.csv, the VTM chunk layout is taken from
the manifest (each chunk = an independent bitstream starting with an intra frame; RA chunks are the CTC intra period).
Every chunk is encoded at QP 22/27/32/37 with the settings of eval_png_vvenc_nvenc.py (VVenC medium, GOP 32 RA / 8 LD,
no QPA, internal 10 bit; NVENC p5 constqp, 3 B frames + middle refs RA / no B LD, main10 for 10-bit sources), decoded,
concatenated and scored with eval_ctc_chunks.metrics (PSNR, MS-SSIM, VMAF, VMAF NEG, LPIPS-Alex at native bit depth).
Cache: <out>/<codec>/<preset>/<class>/<sequence>/rate_<qp>.json. Writes <out>/ctc_vvenc_nvenc_sequences.csv.
"""
import argparse, csv, json, os, shutil, subprocess, tempfile
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eval_ctc_chunks import METRICS, metrics  # noqa: E402

HOME = Path.home()
VE, VD = HOME/'.local/bin/vvencapp', HOME/'.local/bin/vvdecapp'
QPS = [22, 27, 32, 37]


def encode_decode(a, codec, preset, raw, n, period, w, h, depth, qp, tmp):
    """Encode one chunk and decode it to raw YUV at the source bit depth; returns (bitstream bytes, decoded path)."""
    fmt = 'yuv420p10le' if depth > 8 else 'yuv420p'
    bs, dec = tmp/'stream.bin', tmp/'dec.yuv'
    if codec == 'vvenc':
        extra = f'GOPSize={8 if preset == "ld" else 32}:PicReordering={0 if preset == "ld" else 1}:POC0IDR=1:InternalBitDepth=10'
        enc = [VE, '-i', raw, '-s', f'{w}x{h}', '--fps', '30', '--format', 'yuv420_10' if depth > 8 else 'yuv420',
               '--frames', n, '--preset', 'medium', '--threads', a.threads, '--qpa', '0', '--qp', qp,
               '--intraperiod', period, '--refreshtype', 'none' if preset == 'ld' else 'idr', '--additional', extra, '-o', bs]
        y4m = tmp/'dec.y4m'
        steps = [enc, [VD, '-b', bs, '-o', y4m, '-t', a.threads],
                 [a.ffmpeg, '-nostdin', '-y', '-v', 'error', '-i', y4m, '-pix_fmt', fmt, '-f', 'rawvideo', dec]]
    else:
        ten = ['-profile:v', 'main10', '-pix_fmt', 'p010le'] if depth > 8 else []
        enc = [a.ffmpeg, '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', fmt, '-s', f'{w}x{h}', '-r', '30',
               '-i', raw, '-frames:v', n, '-c:v', 'hevc_nvenc', *ten, '-preset', 'p5', '-rc', 'constqp', '-qp', qp,
               '-g', period, '-bf', '0' if preset == 'ld' else '3', '-rc-lookahead', '0', '-forced-idr', '1',
               '-b_ref_mode', 'disabled' if preset == 'ld' else 'middle', '-zerolatency', '1' if preset == 'ld' else '0',
               '-f', 'hevc', bs]
        steps = [enc, [a.ffmpeg, '-nostdin', '-y', '-v', 'error', '-i', bs, '-pix_fmt', fmt, '-f', 'rawvideo', dec]]
    with open(tmp/'codec.log', 'w') as log:
        for c in steps:
            subprocess.run(list(map(str, c)), stdout=log, stderr=subprocess.STDOUT, check=True)
    assert dec.stat().st_size == w*h*3//2*(2 if depth > 8 else 1)*n, f'{codec}: decoded size mismatch'
    return bs.stat().st_size, dec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--toolkit', type=Path, default=Path(NVC_ROOT, 'codecs/codec_results'))
    p.add_argument('--source', type=Path, default=Path(NVC_ROOT, 'codecs/source_yuv'))
    p.add_argument('--ctc', type=Path, default=Path('results/ctc_metrics'), help='holds the VTM ctc_metrics_sequences.csv')
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics/vvenc_nvenc'))
    p.add_argument('--tmp', type=Path, default=Path(NVC_ROOT, 'codecs/tmp_vvenc_nvenc'))
    p.add_argument('--codecs', nargs='+', default=['vvenc', 'nvenc'], choices=['vvenc', 'nvenc'])
    p.add_argument('--presets', nargs='+', default=['ra', 'ld'])
    p.add_argument('--sequence', help='only sequences whose name contains this string')
    p.add_argument('--matrix', default='bt709')
    p.add_argument('--threads', type=int, default=4, help='VVenC/vvdec threads per chunk (also VMAF threads)')
    p.add_argument('--jobs', type=int, default=5, help='chunks encoded in parallel')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ffmpeg', default=str(HOME/'.local/ffmpeg-nvenc/bin/ffmpeg'))
    p.add_argument('--vmaf-ffmpeg', default=str(HOME/'opt/ffmpeg-vmaf/bin/ffmpeg'))
    a = p.parse_args()
    os.environ['LD_LIBRARY_PATH'] = f"{HOME/'.local/ffmpeg-nvenc/lib'}:/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    cfg = json.loads((a.toolkit/'dataset_config.json').read_text())['test_classes']
    vtm = {(r['preset'], r['cls'], r['sequence']) for r in csv.DictReader((a.ctc/'ctc_metrics_sequences.csv').open())
           if r['codec'] == 'vtm' and r['preset'] in a.presets and (not a.sequence or a.sequence in r['sequence'])}
    layout = defaultdict(list)  # VTM chunks of the lowest QP (same layout at every QP)
    for r in csv.DictReader((a.toolkit/'manifest.csv').open()):
        k = (r['preset'], r['sequence_class'], r['sequence'])
        if r['codec'] == 'vtm' and r['rate'] == 'rate_22' and k in vtm:
            layout[k].append(r)
    jobs = [(c, k) for c in a.codecs for k in sorted(layout)]
    print(f'{len(layout)} sequence/preset layouts x {len(a.codecs)} codecs x {len(QPS)} QPs', flush=True)

    import lpips
    alex = lpips.LPIPS(net='alex', verbose=False).eval().to(a.device)
    a.tmp.mkdir(parents=True, exist_ok=True)
    for i, (codec, (preset, cls, seq)) in enumerate(jobs, 1):
        rows = sorted(layout[(preset, cls, seq)], key=lambda r: int(r['first_frame']))
        w, h, depth = int(rows[0]['width']), int(rows[0]['height']), int(rows[0]['bitdepth'])
        first, n = int(rows[0]['first_frame']), sum(int(r['frames']) for r in rows)
        period = max(int(r['frames']) for r in rows)
        src = a.source/cfg[cls]['base_path']/f'{seq}.yuv'
        fbytes = w*h*3//2*(2 if depth > 8 else 1)
        for qp in QPS:
            out = a.out/codec/preset/cls/seq/f'rate_{qp}.json'
            if out.exists():
                continue
            tmp = Path(tempfile.mkdtemp(dir=a.tmp))
            try:
                def one(j, r):  # each chunk is an independent bitstream: encode chunks in parallel
                    d = tmp/f'c{j:05}'; d.mkdir()
                    with open(src, 'rb') as s:
                        s.seek(int(r['first_frame'])*fbytes)
                        (d/'chunk.yuv').write_bytes(s.read(int(r['frames'])*fbytes))
                    b, dec = encode_decode(a, codec, preset, d/'chunk.yuv', int(r['frames']), period, w, h, depth, qp, d)
                    (d/'chunk.yuv').unlink()
                    return b
                with ThreadPoolExecutor(a.jobs) as ex:
                    nbytes = sum(ex.map(one, range(len(rows)), rows))
                recon = tmp/'recon.yuv'
                with open(recon, 'wb') as rec:
                    for j in range(len(rows)):
                        with open(tmp/f'c{j:05}'/'dec.yuv', 'rb') as f:
                            shutil.copyfileobj(f, rec)
                        shutil.rmtree(tmp/f'c{j:05}')
                fr = metrics(a, src, recon, w, h, depth, first, n, alex, tmp)
                res = dict(codec=codec, preset=preset, cls=cls, sequence=seq, rate=f'rate_{qp}', width=w, height=h,
                           bitdepth=depth, first_frame=first, frames=n, chunks=len(rows), bytes=nbytes, bpp=nbytes*8/(w*h*n),
                           **{m: float(np.nanmean([f[m] for f in fr])) for m in METRICS})
                out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
                print(f"[{i}/{len(jobs)}] {codec} {preset} {cls} {seq} QP{qp}  bpp {res['bpp']:.4f}  "
                      f"PSNR-Y {res['psnr_y']:.2f}  VMAF {res['vmaf']:.1f}  LPIPS {res['lpips_alex']:.4f}", flush=True)
            except Exception as e:
                print(f'[{i}/{len(jobs)}] FAIL {codec} {preset} {cls} {seq} QP{qp}: {e} (log: {tmp}/codec.log)', flush=True)
                continue
            shutil.rmtree(tmp)

    rows = [json.loads(f.read_text()) for f in sorted(a.out.glob('*/*/*/*/rate_*.json'))]
    if rows:
        with (a.out/'ctc_vvenc_nvenc_sequences.csv').open('w', newline='') as f:
            wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"wrote {a.out/'ctc_vvenc_nvenc_sequences.csv'} ({len(rows)} rows)", flush=True)


if __name__ == '__main__':
    main()
