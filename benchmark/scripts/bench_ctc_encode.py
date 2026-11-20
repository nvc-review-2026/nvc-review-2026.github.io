#!/usr/bin/env python3
"""Encoder throughput / peak RAM on one GoP of 1080p (run alone, nothing else on the machine).

One encoder process at a time, single-threaded, on the same sequences as bench_ctc_decode.py.
GoP = 32 frames for RA, 8 frames for LD (the GOP sizes of the VTM/ECM CTC configs; LD also matches the Isambard chunks).
enc_fps = frames / encoder wall time; peak_ram_mb = max RSS (/usr/bin/time). Bitstreams go to a temp dir and are deleted.
Encoder flags follow the JVET / AOM CTC; edit FLAGS if the Isambard job scripts used different ones.
Writes <out>/ctc_encode_bench.csv and ctc_encode_bench_summary.csv; reruns skip points already in the csv.
"""
import os
import argparse, csv, subprocess, tempfile
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from collections import defaultdict
from pathlib import Path

E = Path(NVC_ROOT, 'codecs/encoders')
SRC = Path(NVC_ROOT, 'codecs/source_yuv')
SEQS = [('AOM-CTC-A2', SRC/'AOM-CTC/a2_2k/Aerial3200_1920x1080_5994_10bit_420.yuv', 10, 60),
        ('MCL-JCV', SRC/'MCL-JCV/1920x1080/videoSRC10_1920x1080_yuv420p.yuv', 8, 30)]
RATES = {'vtm': [22, 27, 32, 37], 'ecm': [22, 27, 32, 37], 'av1': [20, 32, 43, 55], 'avm': [20, 32, 43, 55]}
GOP = {'ra': 32, 'ld': 8}
AOM_COMMON = ['--passes=1', '--end-usage=q', '--cpu-used=0', '--threads=1', '--tile-columns=0',
              '--use-fixed-qp-offsets=1', '--deltaq-mode=0', '--enable-tpl-model=0', '--min-gf-interval=16', '--max-gf-interval=16']
FLAGS = {'ra': ['--lag-in-frames=19', '--auto-alt-ref=1', '--gf-min-pyr-height=4', '--gf-max-pyr-height=4',
                '--kf-min-dist=65', '--kf-max-dist=65'],
         'ld': ['--lag-in-frames=0', '--gf-min-pyr-height=0', '--gf-max-pyr-height=0',
                '--kf-min-dist=9999', '--kf-max-dist=9999']}


def cmd(codec, preset, src, depth, fps, q, n, bs):
    if codec in ('vtm', 'ecm'):
        cfg = E/codec/'cfg'/f"encoder_{'randomaccess' if preset == 'ra' else 'lowdelay'}_{codec}.cfg"
        return [E/codec/'bin/EncoderAppStatic', '-c', cfg, '-i', src, '-b', bs, '-o', '', '-wdt', '1920', '-hgt', '1080',
                '-fr', str(fps), '-f', str(n), '-q', str(q), f'--InputBitDepth={depth}', '--InternalBitDepth=10',
                f'--OutputBitDepth={depth}', '--InputChromaFormat=420']
    enc = next(p for p in (E/codec/'build/avmenc', E/codec/'build/aomenc') if p.exists())
    return [enc, *AOM_COMMON, *FLAGS[preset], f'--cq-level={q}', f'--limit={n}', '--i420', '--width=1920', '--height=1080',
            f'--fps={fps}/1', f'--input-bit-depth={depth}', f'--bit-depth={depth}', '--ivf', '-o', bs, src]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics'))
    p.add_argument('--codecs', nargs='+', default=['vtm', 'av1', 'avm', 'ecm'])
    p.add_argument('--presets', nargs='+', default=['ra', 'ld'])
    p.add_argument('--classes', nargs='+', default=[s[0] for s in SEQS])
    p.add_argument('--rate-index', type=int, nargs='+', default=[0, 1, 2, 3], help='which of the 4 rate points (0 = highest rate)')
    a = p.parse_args()

    out = a.out/'ctc_encode_bench.csv'
    rows = list(csv.DictReader(out.open())) if out.exists() else []
    done = {(r['codec'], r['preset'], r['cls'], r['rate']) for r in rows}
    tmp = Path(tempfile.mkdtemp(dir=a.out))
    for codec in a.codecs:
        for preset in a.presets:
            for cls, src, depth, fps in (q for q in SEQS if q[0] in a.classes):
                for q in (RATES[codec][i] for i in a.rate_index):
                    if (codec, preset, cls, str(q)) in done:
                        continue
                    n = GOP[preset]
                    c = cmd(codec, preset, src, depth, fps, q, n, tmp/'stream.bin')
                    r = subprocess.run(['/usr/bin/time', '-f', '%e %M', '-o', tmp/'time', *map(str, c)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                    if r.returncode:
                        print(f'FAIL {codec} {preset} {cls} {q}:', r.stderr[-800:], flush=True)
                        continue
                    e, m = (tmp/'time').read_text().split()[-2:]
                    rows.append(dict(codec=codec, preset=preset, cls=cls, rate=q, frames=n, enc_seconds=float(e),
                                     enc_fps=n/float(e), peak_ram_mb=int(m)/1024, bytes=(tmp/'stream.bin').stat().st_size))
                    print(f'{codec} {preset} {cls} {q}: {n} frames, {float(e):.1f} s, {n/float(e):.3f} fps, {int(m)/1024:.0f} MB', flush=True)
                    with out.open('w', newline='') as f:
                        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()

    agg = defaultdict(list)
    for r in rows:
        agg[(r['codec'], r['preset'])].append(r)
    with (a.out/'ctc_encode_bench_summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['codec', 'preset', 'points', 'frames', 'enc_fps_1080p', 'sec_per_frame', 'peak_ram_mb_1080p'])
        for (c, pr), v in sorted(agg.items()):
            fr, s = sum(int(r['frames']) for r in v), sum(float(r['enc_seconds']) for r in v)
            w.writerow([c, pr, len(v), fr, fr/s, s/fr, max(float(r['peak_ram_mb']) for r in v)])
    print(f'wrote {out} and ctc_encode_bench_summary.csv', flush=True)


if __name__ == '__main__':
    main()
