#!/usr/bin/env python3
"""Fig. content-diagnostics: per-sequence PSNR-Y BD-rate vs VTM against source SI / TI (LD top, RA bottom).

Same data as siti_profile.py (siti_1080p.csv + bdrate_vs_vtm_sequences.csv + bdrate_learned_sequences.csv), each codec's
own valid sequences, Spearman rho with bootstrap 95% CI in the legend. Only codecs with trusted per-sequence data (GIViC
and NVRC per-sequence BD-rates are not reliable); BD-rates of -100% (no overlapping quality range) are dropped. Writes DVC-Review-PIEEE/figures/content_diagnostics.pdf and the plotted points to DVC-Review-PIEEE/data/content/siti_bdrate.csv.
  python scripts/plot_siti_paper.py              rebuild siti_bdrate.csv from the BD-rate results, then plot
  python scripts/plot_siti_paper.py --from-csv   plot siti_bdrate.csv as it is (edit it to change the figure)
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from siti_profile import ALIAS, spearman_ci  # noqa: E402

SITI = Path('DVC-Review-PIEEE/data/content/siti_1080p.csv')
BD = [Path('results/ctc_metrics/bdrate_vs_vtm_sequences.csv'), Path('results/ctc_metrics/bdrate_learned_sequences.csv')]
OUT = Path('DVC-Review-PIEEE/figures/content_diagnostics.pdf')
CSV = Path('DVC-Review-PIEEE/data/content/siti_bdrate.csv')
CODECS = {'ld': [('ecm', 'ECM', '#2b5d8c'), ('avm', 'AVM', '#8c6bb1'), ('dcvc-rt', 'DCVC-RT', '#E4570F'),
                 ('dcvc-uf', 'DCVC-UF (LD)', '#4FB0A0')],
          'ra': [('ecm', 'ECM', '#2b5d8c'), ('avm', 'AVM', '#8c6bb1'), ('dcvc-uf-hts', 'DCVC-UF (HT-S)', '#2E7D6B')]}


COLOR = {l: c for v in CODECS.values() for _, l, c in v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-csv', action='store_true', help='plot siti_bdrate.csv instead of rebuilding it')
    if ap.parse_args().from_csv:
        plot(list(csv.DictReader(CSV.open())))
        return
    content = {ALIAS.get(r['name'], r['name']): r for r in csv.DictReader(SITI.open())}
    def name(seq):
        return next((n for n in content if seq == n or seq.startswith(n + '_')), None)

    bd = defaultdict(dict)
    for r in (r for f in BD for r in csv.DictReader(f.open())):
        n = name(r['sequence'])
        if n and r['psnr_y'] not in ('', 'nan') and float(r['psnr_y']) > -95:
            bd[(r['preset'], r['test'])][n] = float(r['psnr_y'])

    with CSV.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'codec', 'sequence', 'si', 'ti', 'bdrate_psnr_y_vs_vtm'])
        for preset, codecs in CODECS.items():
            for key, label, _ in codecs:
                for n, b in sorted(bd[(preset, key)].items()):
                    w.writerow([preset.upper(), label, n, content[n]['si8'], content[n]['ti8'], b])
    plot(list(csv.DictReader(CSV.open())))


def plot(rows):
    """rows of siti_bdrate.csv; codecs in order of appearance, colours from CODECS (others: default cycle)."""
    cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    order = {p: list(dict.fromkeys(r['codec'] for r in rows if r['config'] == p.upper())) for p in ('ld', 'ra')}
    cmap = {l: COLOR.get(l) or next(cycle) for v in order.values() for l in v}
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), sharey='row', sharex='col')
    for row, preset in enumerate(['ld', 'ra']):
        for col, (feat, xl) in enumerate([('si8', 'SI'), ('ti8', 'TI')]):
            ax = axes[row, col]
            rhos = []
            for label in order[preset]:
                pts = [r for r in rows if r['config'] == preset.upper() and r['codec'] == label]
                x, y = [float(r[feat[:2]]) for r in pts], [float(r['bdrate_psnr_y_vs_vtm']) for r in pts]
                c = cmap[label]
                rho, lo, hi = spearman_ci(x, y)
                ax.scatter(x, y, s=11, color=c, alpha=.8, lw=.3, edgecolor='white', label=label, zorder=3)
                rhos.append((f'$\\rho$ = {rho:+.2f}', c, lo > 0 or hi < 0))
            for i, (t, color, sig) in enumerate(rhos):  # rho per codec, bold when the 95% CI excludes 0
                ax.text(.02, .97 - i * .085, t, transform=ax.transAxes, color=color, fontsize=7, va='top',
                        fontweight='bold' if sig else 'normal',
                        bbox=dict(fc='white', ec='none', alpha=.7, pad=.5))
            ax.axhline(0, color='0.35', lw=.6, zorder=1)
            ax.set_yscale('symlog', linthresh=100, linscale=1.5)
            ax.set_yticks([-100, -50, 0, 50, 100, 1000] if preset == 'ra' else [-100, -50, 0, 50, 100])
            ax.set_yticklabels([f'{t:g}' for t in ax.get_yticks()])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(color='0.92', lw=.5, zorder=0)
            if row == 1:
                ax.set_xlabel(f'Source {xl}')
            if col == 0:
                ax.set_ylabel(f'BD-rate vs VTM {preset.upper()} (%)')
            ax.set_title(f'({"abcd"[2*row+col]}) {preset.upper()}, {xl}', fontsize=8, loc='left')
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh); l.append(ll)
    fig.tight_layout(h_pad=.8, w_pad=.6, rect=(0, 0, 1, .95))
    fig.legend(h, l, fontsize=7.5, frameon=False, loc='upper center', ncol=len(l), bbox_to_anchor=(.5, 1),
               handletextpad=.1, columnspacing=1.2, markerscale=1.4)
    fig.savefig(OUT, bbox_inches='tight')
    fig.savefig(OUT.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
