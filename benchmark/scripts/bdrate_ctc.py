#!/usr/bin/env python3
"""BD-rate of ECM / AV1 / AVM vs VTM (RA vs RA, LD vs LD) from ctc_metrics_sequences.csv.

Per sequence: only sequences with all 4 rate points for both the test codec and VTM in that preset,
and (unless --any-chunks) the same number of independently coded chunks.
BD-rate with piecewise-cubic (PCHIP) interpolation of log(bpp) over the overlapping quality range (VCEG-AM / JVET style).
Quality axes: PSNR-Y, PSNR-YUV (6:1:1), MS-SSIM-Y as -10*log10(1-MS-SSIM) dB, VMAF, LPIPS-Alex as -LPIPS (lower is better).
Negative = bitrate saving vs VTM. Sequences whose curves do not overlap on a metric are skipped for that metric.
Writes <out>/bdrate_vs_vtm_sequences.csv and bdrate_vs_vtm_datasets.csv (mean per dataset and overall).
"""
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

METRICS = {'psnr_y': lambda r: float(r['psnr_y']),
           'psnr_yuv': lambda r: (6*float(r['psnr_y']) + float(r['psnr_u']) + float(r['psnr_v'])) / 8,
           'msssim_y': lambda r: -10*math.log10(max(1 - float(r['msssim_y']), 1e-10)),
           'vmaf': lambda r: float(r['vmaf']),
           'lpips_alex': lambda r: -float(r['lpips_alex'])}
DATASET = {'HEVC_B': 'HEVC-CTC', 'HEVC_C': 'HEVC-CTC', 'HEVC_D': 'HEVC-CTC', 'HEVC_E': 'HEVC-CTC',
           'AOM-CTC-A2': 'AOM-CTC', 'AOM-CTC-A3': 'AOM-CTC', 'AOM-CTC-A4': 'AOM-CTC', 'AOM-CTC-A5': 'AOM-CTC'}


def bdrate(r1, q1, r2, q2):
    """Average % rate difference of curve 2 vs curve 1 (anchor) over the common quality interval."""
    lo, hi = max(min(q1), min(q2)), min(max(q1), max(q2))
    if lo >= hi:
        return float('nan')
    def area(r, q):
        o = np.argsort(q)
        q, lr = np.asarray(q)[o], np.log10(np.asarray(r)[o])
        if np.any(np.diff(q) <= 0):
            return None
        return PchipInterpolator(q, lr).integrate(lo, hi)
    a1, a2 = area(r1, q1), area(r2, q2)
    if a1 is None or a2 is None:
        return float('nan')
    return (10 ** ((a2 - a1) / (hi - lo)) - 1) * 100


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=Path, nargs='+', default=[Path('results/ctc_metrics/ctc_metrics_sequences.csv')])
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics'))
    p.add_argument('--anchor', default='vtm')
    p.add_argument('--any-chunks', action='store_true', help='also keep sequences chunked differently from the anchor')
    p.add_argument('--tests', nargs='+', default=['ecm', 'av1', 'avm'])
    a = p.parse_args()

    pts = defaultdict(list)
    for r in (r for c in a.csv for r in csv.DictReader(c.open())):
        pts[(r['codec'], r['preset'], r['cls'], r['sequence'])].append(r)
    rows = []
    for (codec, preset, cls, seq), test in sorted(pts.items()):
        anchor = pts.get((a.anchor, preset, cls, seq), [])
        if codec not in a.tests or len(test) != 4 or len(anchor) != 4:
            continue
        # Every chunk is an independent bitstream starting with an intra frame, so only compare equal chunking.
        if not a.any_chunks and {r['chunks'] for r in test} != {r['chunks'] for r in anchor}:
            continue
        row = dict(test=codec, preset=preset, dataset=DATASET.get(cls, cls), cls=cls, sequence=seq)
        for m, f in METRICS.items():
            row[m] = bdrate([float(r['bpp']) for r in anchor], [f(r) for r in anchor],
                            [float(r['bpp']) for r in test], [f(r) for r in test])
        rows.append(row)

    with (a.out/'bdrate_vs_vtm_sequences.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['test'], r['preset'], r['dataset'])].append(r)
        groups[(r['test'], r['preset'], 'Overall')].append(r)
    out = []
    for (t, pr, ds), v in sorted(groups.items()):
        d = dict(test=t, preset=pr, dataset=ds, sequences=len(v))
        for m in METRICS:
            x = [r[m] for r in v if not math.isnan(r[m])]
            d[m] = sum(x)/len(x) if x else float('nan'); d[f'{m}_n'] = len(x)
        out.append(d)
    with (a.out/'bdrate_vs_vtm_datasets.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)

    for pr in ('ra', 'ld'):
        print(f"\n### BD-rate vs VTM ({pr.upper()}), %  (negative = saving)\n"
              "| Codec | Dataset | #Seq | PSNR-Y | PSNR-YUV | MS-SSIM-Y | VMAF | LPIPS-Alex |\n|---|---|---|---|---|---|---|---|")
        for d in out:
            if d['preset'] == pr:
                print(f"| {d['test'].upper()} | {d['dataset']} | {d['sequences']} | " +
                      ' | '.join(f"{d[m]:+.2f}" for m in METRICS) + ' |')
    print(f"\nwrote {a.out/'bdrate_vs_vtm_sequences.csv'} and bdrate_vs_vtm_datasets.csv")


if __name__ == '__main__':
    main()
