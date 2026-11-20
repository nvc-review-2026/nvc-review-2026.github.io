#!/usr/bin/env python3
"""BD-rate of learned codecs (paper repo rd_points.csv) vs our VTM (ctc_metrics_sequences.csv), same output format as
bdrate_ctc.py so siti_profile.py can read both.

LD codecs vs VTM LD, RA codecs vs VTM RA. 8-bit sources only: the learned runs code 8-bit YUV, while the VTM anchor of
the 10-bit AOM sequences is scored on the native 10-bit source, so those PSNRs are not comparable.
Learned codecs run the full sequence with their own intra period (not the VTM chunking) - see the paper caption.
Writes <out>/bdrate_learned_sequences.csv.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from bdrate_ctc import METRICS, bdrate  # noqa: E402

# (codec, config, variant, source) in rd_points.csv -> output name, VTM preset
LEARNED = {('DCVC-RT', 'LD', '', 'dvc-bench-runs'): ('dcvc-rt', 'ld'),
           ('DCVC-UF', 'LD', '', 'dvc-bench-runs'): ('dcvc-uf', 'ld'),
           ('DCVC-UF', 'RA', 'HT-S', 'dvc-bench-runs'): ('dcvc-uf-hts', 'ra'),
           ('GIViC', 'RA', '', 'NVC-data'): ('givic', 'ra'),
           ('NVRC', 'RA', 'seg', 'dvc-bench-runs'): ('nvrc', 'ra')}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rd', type=Path, default=Path('DVC-Review-PIEEE/data/rd/rd_points.csv'))
    p.add_argument('--vtm', type=Path, default=Path('results/ctc_metrics/ctc_metrics_sequences.csv'))
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics'))
    a = p.parse_args()

    vtm = defaultdict(list)
    for r in csv.DictReader(a.vtm.open()):
        if r['codec'] == 'vtm' and int(r['bitdepth']) == 8:
            vtm[(r['preset'], r['sequence'])].append(r)
    names = {s for _, s in vtm}
    def match(n):  # rd_points names may lack the bitstream suffix
        return n if n in names else next((s for s in names if s.startswith(n + '_')), None)

    test = defaultdict(list)
    for r in csv.DictReader(a.rd.open()):
        k = LEARNED.get((r['codec'], r['config'], r['variant'], r['source']))
        if k and match(r['name']) and r['psnr_y']:
            r['psnr_u'] = r['psnr_u'] or r['psnr_y']; r['psnr_v'] = r['psnr_v'] or r['psnr_y']  # psnr_yuv needs them
            test[(*k, r['cls'], match(r['name']))].append(r)

    rows = []
    for (codec, preset, cls, seq), t in sorted(test.items()):
        anc = vtm.get((preset, seq), [])
        if len(anc) != 4 or len(t) < 4:
            continue
        row = dict(test=codec, preset=preset, dataset=cls, cls=cls, sequence=seq)
        for m, f in METRICS.items():
            try:
                row[m] = bdrate([float(r['bpp']) for r in anc], [f(r) for r in anc],
                                [float(r['bpp']) for r in t], [f({**r, 'lpips_alex': r['lpips']}) for r in t])
            except (ValueError, KeyError):
                row[m] = float('nan')
        rows.append(row)
    out = a.out/'bdrate_learned_sequences.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    for c in sorted({r['test'] for r in rows}):
        v = [r['psnr_y'] for r in rows if r['test'] == c and r['psnr_y'] == r['psnr_y']]
        print(f'{c}: {len(v)} sequences, mean PSNR-Y BD-rate vs VTM {sum(v)/len(v):+.1f}%')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
