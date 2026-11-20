#!/usr/bin/env python3
"""Per-sequence BD-rate vs VTM against content SI / TI on the landscape 1080p set (siti_1080p.csv).

For each preset (LD, RA) and codec: scatter BD-rate (PSNR-Y by default) vs SI and vs TI, marker shape = frame-rate group,
Spearman rho with a bootstrap 95% CI. Two sequence sets: each codec's own valid sequences ('own') and the sequences
valid for all codecs of that preset ('common'), the latter as the robustness check.
Writes <out>/siti_profile_<metric>.png and siti_profile_<metric>.csv.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALIAS = {'ReadySteadyGo': 'ReadySetGo'}
FPS_GROUPS = [('≤30 fps', 0, 31, 'o'), ('50/60 fps', 31, 61, 's'), ('120 fps', 61, 1e9, '^')]


def spearman_ci(x, y, n=2000, seed=0):
    rho = spearmanr(x, y)[0]
    rng, x, y = np.random.default_rng(seed), np.asarray(x), np.asarray(y)
    bs = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return rho, *np.nanpercentile(bs, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--siti', type=Path, default=Path('DVC-Review-PIEEE/data/content/siti_1080p.csv'))
    p.add_argument('--bd', type=Path, nargs='+', default=[Path('results/ctc_metrics/bdrate_vs_vtm_sequences.csv')])
    p.add_argument('--metric', default='psnr_y')
    p.add_argument('--tests', nargs='+', default=['ecm', 'av1', 'avm'])
    p.add_argument('--out', type=Path, default=Path('results/siti_profile'))
    a = p.parse_args()

    content = {ALIAS.get(r['name'], r['name']): r for r in csv.DictReader(a.siti.open())}
    def name(seq):  # bitstream names carry suffixes such as _1920x1080_24
        return next((n for n in content if seq == n or seq.startswith(n + '_')), None)

    bd = defaultdict(dict)  # (preset, codec) -> {sequence: bd-rate}
    for r in (r for f in a.bd for r in csv.DictReader(f.open())):
        n = name(r['sequence'])
        if n and r['test'] in a.tests and r[a.metric] not in ('', 'nan'):
            bd[(r['preset'], r['test'])][n] = float(r[a.metric])

    a.out.mkdir(parents=True, exist_ok=True)
    stats = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey='row')
    for row, preset in enumerate(['ld', 'ra']):
        codecs = [c for c in a.tests if bd[(preset, c)]]
        common = set.intersection(*(set(bd[(preset, c)]) for c in codecs)) if codecs else set()
        for col, feat in enumerate(['si8', 'ti8']):
            ax = axes[row, col]
            for k, c in enumerate(codecs):
                d = bd[(preset, c)]
                for label, lo, hi, mk in FPS_GROUPS:
                    s = [n for n in d if lo <= float(content[n]['fps']) < hi]
                    ax.scatter([float(content[n][feat]) for n in s], [d[n] for n in s], marker=mk, color=f'C{k}',
                               s=22, alpha=.75, label=f'{c.upper()}' if mk == 'o' else None)
                for subset, seqs in (('own', list(d)), ('common', sorted(common))):
                    if len(seqs) < 4:
                        continue
                    rho, lo, hi = spearman_ci([float(content[n][feat]) for n in seqs], [d[n] for n in seqs])
                    stats.append(dict(preset=preset, codec=c, feature=feat[:2].upper(), subset=subset, n=len(seqs),
                                      rho=rho, ci_lo=lo, ci_hi=hi))
            ax.axhline(0, color='grey', lw=.6)
            ax.set_xlabel(f'{feat[:2].upper()} (8-bit scale, native frame rate)')
            ax.set_ylabel(f'{preset.upper()} BD-rate vs VTM, {a.metric} (%)' if col == 0 else '')
            ax.set_title(f'{preset.upper()}: BD-rate vs {feat[:2].upper()}  (common set n={len(common)})', fontsize=10)
        axes[row, 0].legend(fontsize=8)
    for label, _, _, mk in FPS_GROUPS:  # frame-rate key
        axes[0, 1].scatter([], [], marker=mk, color='k', label=label)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out/f'siti_profile_{a.metric}.png', dpi=150)

    with (a.out/f'siti_profile_{a.metric}.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(stats[0])); w.writeheader(); w.writerows(stats)
    print('| Preset | Codec | Feature | Set | n | Spearman rho [95% CI] |\n|---|---|---|---|---|---|')
    for s in stats:
        print(f"| {s['preset'].upper()} | {s['codec'].upper()} | {s['feature']} | {s['subset']} | {s['n']} | "
              f"{s['rho']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}] |")
    print(f"wrote {a.out/f'siti_profile_{a.metric}.png'} and .csv")


if __name__ == '__main__':
    main()
