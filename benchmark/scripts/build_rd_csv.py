#!/usr/bin/env python3
"""RD data behind the paper's RD figures, two editable csvs in DVC-Review-PIEEE/data/rd/:
  rd_curves.csv     (row = avg) the dataset-average point that plot_paper_figures.py draws: mean bpp and mean quality per rate point over the
       sequences that every codec drawn in that (config, dataset) has (8-bit sources only; a codec covering
       < --min-cover of the dataset's sequences gets no avg rows). n_seq = number of sequences averaged.
  rd_sequences.csv  (row = seq) one sequence x rate point of one codec (config LD / RA), with its source file
Column `use` (1/0): set 0 to leave a row out. In rd_sequences.csv it drops the point from the averages, in rd_curves.csv
from the figure. Editing rd_curves.csv changes the figure directly; edit rd_sequences.csv and run with --average.

  python scripts/build_rd_csv.py             rebuild both csvs from the result files (overwrites manual edits)
  python scripts/build_rd_csv.py --average   keep rd_sequences.csv as it is, recompute only rd_curves.csv

Sources: conventional codecs = results/ctc_metrics/ctc_metrics_sequences.csv and vvenc_nvenc/ctc_vvenc_nvenc_sequences.csv
(same chunks as VTM only); DCVC-DC / DCVC-FM = results/dcvc_metrics/dcvc-{dc,fm}/*/*/rate_*.json (one I frame, full
sequence); other learned codecs = data/rd/rd_points.csv (source as in the figure spec).
"""
import argparse, csv, glob, json, math
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_paper_figures import LD_ALL, RA_ALL, REPO, dataset_of, key  # noqa: E402

OUT = REPO/'data/rd/rd_curves.csv'
SEQ = REPO/'data/rd/rd_sequences.csv'
Q = ['psnr_y', 'msssim_y', 'vmaf', 'lpips']
FIELDS = ['row', 'use', 'config', 'codec', 'family', 'dataset', 'sequence', 'rate', 'n_seq', 'bpp', *Q, 'source']
SPECS = {'LD': LD_ALL, 'RA': RA_ALL}  # label -> (source, selector, family)
OFF = {'AVM'}  # use = 0 by default: wrong encoder configuration


def from_sources(a):
    rows, tenbit = [], set()
    ctc = list(csv.DictReader(a.ctc.open()))
    vtm = {(r['preset'], r['sequence']): (r['chunks'], r['frames']) for r in ctc if r['codec'] == 'vtm'}
    for src, f in (('ctc', a.ctc), ('ctc2', a.ctc2)):
        for r in (ctc if src == 'ctc' else csv.DictReader(f.open())):
            if int(r['bitdepth']) > 8:
                tenbit.add(key(r['sequence']))
                continue
            for cfg, spec in SPECS.items():
                lab = next((l for l, (s, sel, _) in spec.items() if s == src and sel == (r['codec'], r['preset'])), None)
                if lab and vtm.get((r['preset'], r['sequence'])) == (r['chunks'], r['frames']):
                    rows.append(dict(config=cfg, codec=lab, family=spec[lab][2], dataset=dataset_of(r['cls']),
                                     sequence=key(r['sequence']), rate=r['rate'], bpp=r['bpp'], psnr_y=r['psnr_y'],
                                     msssim_y=r['msssim_y'], vmaf=r['vmaf'], lpips=r['lpips_alex'], source=f.name))
    for r in csv.DictReader(a.rd.open()):
        if r['src_format'] != '8bit':
            continue
        for cfg, spec in SPECS.items():
            lab = next((l for l, (s, sel, _) in spec.items()
                        if s == 'rd' and sel == (r['codec'], r['config'], r['variant'], r['source'])), None)
            if not lab or (lab == 'GIViC' and dataset_of(r['cls']) not in ('UVG', 'MCL-JCV')):
                continue  # the AOM / HEVC GIViC exports are not GIViC results
            rows.append(dict(config=cfg, codec=lab, family=spec[lab][2], dataset=dataset_of(r['cls']),
                             sequence=key(r['name']), rate=r['rate'], bpp=r['bpp'], **{m: r[m] for m in Q},
                             source=f"rd_points.csv:{r['source']}"))
    for f in sorted(glob.glob(str(a.dcvc/'dcvc-[df][cm]'/'*'/'*'/'rate_*.json'))):
        r = json.loads(Path(f).read_text())
        lab = {'dcvc-dc': 'DCVC-DC', 'dcvc-fm': 'DCVC-FM'}[r['model']]
        rows.append(dict(config='LD', codec=lab, family='agnostic', dataset=dataset_of(r['cls']), sequence=key(r['sequence']),
                         rate=r['rate'], bpp=r['bpp'], psnr_y=r['psnr_y'], msssim_y=r['msssim_y'], vmaf=r['vmaf'],
                         lpips=r['lpips_alex'], source='dcvc_metrics'))
    rows = [r for r in rows if r['sequence'] not in tenbit]
    for r in rows:
        r.update(row='seq', use=0 if r['codec'] in OFF else 1, n_seq='')
    return rows


def averages(seq, min_cover):
    """avg rows: per (config, dataset) the codecs covering >= min_cover of its sequences, averaged over their common set."""
    pts = defaultdict(lambda: defaultdict(dict))  # (cfg, ds, codec) -> seq -> rate -> row
    fam = {}
    for r in seq:
        if str(r['use']) == '1' and r['bpp'] not in ('', 'nan') and r['psnr_y'] not in ('', 'nan'):
            pts[(r['config'], r['dataset'], r['codec'])][r['sequence']][r['rate']] = r
            fam[(r['config'], r['codec'])] = r['family']
    out = []
    for cfg, ds in sorted({(c, d) for c, d, _ in pts}):
        have = {lab: set(v) for (c, d, lab), v in pts.items() if (c, d) == (cfg, ds)}
        allseq = set().union(*have.values())
        drawn = [l for l, s in have.items() if len(s) >= min_cover * len(allseq)]
        common = set.intersection(*(have[l] for l in drawn))
        for lab in drawn:
            by_rate = defaultdict(list)
            for s in common:
                for rt, r in pts[(cfg, ds, lab)][s].items():
                    by_rate[rt].append(r)
            for rt, v in sorted(by_rate.items()):
                if len(v) != len(common):
                    continue
                mean = lambda m: sum(float(r[m]) for r in v) / len(v) if all(r[m] not in ('', 'nan') for r in v) else math.nan
                out.append(dict(row='avg', use=1, config=cfg, codec=lab, family=fam[(cfg, lab)], dataset=ds, sequence='',
                                rate=rt, n_seq=len(common), bpp=mean('bpp'), **{m: mean(m) for m in Q},
                                source=f'mean of {len(common)} common sequences'))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ctc', type=Path, default=Path('results/ctc_metrics/ctc_metrics_sequences.csv'))
    p.add_argument('--ctc2', type=Path, default=Path('results/ctc_metrics/vvenc_nvenc/ctc_vvenc_nvenc_sequences.csv'))
    p.add_argument('--rd', type=Path, default=REPO/'data/rd/rd_points.csv')
    p.add_argument('--dcvc', type=Path, default=Path('results/dcvc_metrics'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--min-cover', type=float, default=0.5)
    p.add_argument('--seq-out', type=Path, default=SEQ)
    p.add_argument('--average', action='store_true', help='recompute --out from the rows already in --seq-out')
    a = p.parse_args()
    seq = list(csv.DictReader(a.seq_out.open())) if a.average else from_sources(a)
    avg = averages(seq, a.min_cover)
    for f, rows in ((a.out, avg), *(() if a.average else ((a.seq_out, seq),))):
        with f.open('w', newline='') as fh:
            w = csv.DictWriter(fh, FIELDS, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        print(f'wrote {f}: {len(rows)} rows')
    for cfg in ('LD', 'RA'):
        for ds in sorted({r['dataset'] for r in avg if r['config'] == cfg}):
            n = {r['codec']: r['n_seq'] for r in avg if (r['config'], r['dataset']) == (cfg, ds)}
            print(f'  {cfg} {ds}: n={next(iter(n.values()))} common sequences; ' + ', '.join(n))


if __name__ == '__main__':
    main()
