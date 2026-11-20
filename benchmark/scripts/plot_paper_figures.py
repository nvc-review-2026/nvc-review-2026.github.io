#!/usr/bin/env python3
"""Paper figures: RD curves (LD and RA, 4 datasets x PSNR / MS-SSIM / VMAF / LPIPS) and the rate-distortion-complexity
figure that replaces the radar plot.

RD curves: the avg rows (use = 1) of data/rd/rd_curves.csv, written by scripts/build_rd_csv.py (dataset-average
curve of each codec over the sequences common to the codecs of that row; see that script). Codecs with several model
sizes (HiNeRV, NVRC) show the upper envelope of their averaged points.
Each row's bpp axis spans the VTM rate range widened by --xpad; points outside it are not drawn.
Radar (Fig. 1): encoding / decoding FPS (--complexity csv, the complexity table) and the PSNR / MS-SSIM / VMAF BD-rates
on HEVC B-E and UVG (tables/table_ra.tex), LD and RA side by side, each axis normalised so that larger is better.
Writes <out>/rd_lowdelay_v2.pdf, rd_randomaccess_v2.pdf, radar_v2.pdf.
"""
import argparse, csv, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path('DVC-Review-PIEEE')
DATASETS = ['UVG', 'MCL-JCV', 'HEVC B-E', 'AOM A2-A5']
METRICS = [('psnr_y', 'PSNR-Y (dB)'), ('msssim_y', 'MS-SSIM$_Y$'), ('vmaf', 'VMAF'), ('lpips', 'LPIPS $\\downarrow$')]
# label -> (source, selector, family). family: conv / agnostic / adaptive
LD = {'VTM': ('ctc', ('vtm', 'ld'), 'conv'), 'AV1': ('ctc', ('av1', 'ld'), 'conv'), 'ECM': ('ctc', ('ecm', 'ld'), 'conv'),
      'AVM': ('ctc', ('avm', 'ld'), 'conv'),
      'MaskCRT': ('rd', ('MaskCRT', 'LD', '', 'dvc-bench-runs'), 'agnostic'),
      'DCVC-RT': ('rd', ('DCVC-RT', 'LD', '', 'dvc-bench-runs'), 'agnostic'),
      'DCVC-UF (LD)': ('rd', ('DCVC-UF', 'LD', '', 'dvc-bench-runs'), 'agnostic')}
RA = {'VTM': ('ctc', ('vtm', 'ra'), 'conv'), 'AV1': ('ctc', ('av1', 'ra'), 'conv'), 'ECM': ('ctc', ('ecm', 'ra'), 'conv'),
      'AVM': ('ctc', ('avm', 'ra'), 'conv'),
      'B-CANF': ('rd', ('B-CANF', 'RA', '', 'dvc-bench-runs'), 'agnostic'),
      'BRHVC': ('rd', ('BRHVC', 'RA', '', 'dvc-bench-runs'), 'agnostic'),
      'DCVC-UF (HT-S)': ('rd', ('DCVC-UF', 'RA', 'HT-S', 'dvc-bench-runs'), 'agnostic'),
      'HiNeRV': ('rd', ('HiNeRV', 'RA', 'full', 'dvc-bench-runs'), 'adaptive'),
      'NVRC': ('rd', ('NVRC', 'RA', 'seg', 'dvc-bench-runs'), 'adaptive'),
      'PNVC': ('rd', ('PNVC', 'RA', '', 'NVC-data'), 'adaptive'),
      'GIViC': ('rd', ('GIViC', 'RA', '', 'NVC-data'), 'adaptive')}
# + codecs run since; build_rd_csv.py exports all of them, the RD figures draw the avg rows of its csv
LD_ALL = {**LD, 'VVenC': ('ctc2', ('vvenc', 'ld'), 'conv'), 'NVENC': ('ctc2', ('nvenc', 'ld'), 'conv'),
          'DCVC-DC': ('dcvc', 'dcvc-dc', 'agnostic'), 'DCVC-FM': ('dcvc', 'dcvc-fm', 'agnostic')}
RA_ALL = {**RA, 'VVenC': ('ctc2', ('vvenc', 'ra'), 'conv'), 'NVENC': ('ctc2', ('nvenc', 'ra'), 'conv')}
# one colour / marker per codec in every figure
STYLE = {'VTM': ('#000000', 's'), 'AV1': ('#7f7f7f', 'D'), 'ECM': ('#d62728', '^'), 'AVM': ('#8c564b', 'v'),
         'VVenC': ('#bcbd22', 'P'), 'NVENC': ('#17becf', 'X'),
         'MaskCRT': ('#ff7f0e', 'o'), 'DCVC-DC': ('#aec7e8', 'o'), 'DCVC-FM': ('#9edae5', 'o'),
         'DCVC-RT': ('#1f77b4', 'o'), 'DCVC-UF (LD)': ('#2ca02c', 'o'), 'DCVC-UF (HT-S)': ('#006d2c', 's'),
         'B-CANF': ('#e377c2', 'o'), 'BRHVC': ('#9467bd', 'o'),
         'HiNeRV': ('#8c6d31', '*'), 'NVRC': ('#393b79', '*'), 'PNVC': ('#e7298a', '*'), 'GIViC': ('#66a61e', '*')}
LINE = {'conv': '--', 'agnostic': '-', 'adaptive': '-.'}


def dataset_of(cls):
    c = cls.upper().replace('_', '-')
    if c.startswith('UVG'): return 'UVG'
    if c.startswith('MCL'): return 'MCL-JCV'
    if c.startswith('HEVC') or c.startswith('JVET'): return 'HEVC B-E'
    if c.startswith('AOM'): return 'AOM A2-A5'


def key(name):  # 'videoSRC10_1920x1080_yuv420p' and 'videoSRC10' -> 'videosrc10'
    n = re.split(r'_\d{3,4}[x_]\d{3,4}|_\d{3,4}p', name)[0].lower()
    return {'readysteadygo': 'readysetgo'}.get(n, n)


def curves_csv(a, cfg, skip):
    """{(dataset, metric, codec): (upper-envelope points, n_seq)} from the avg rows of the editable csv (build_rd_csv.py)."""
    pts = defaultdict(list)
    for r in csv.DictReader(a.rd_csv.open()):
        if r['row'] == 'avg' and r['use'] == '1' and r['config'] == cfg and r['codec'] not in skip:
            for m, _ in METRICS:
                if r[m] not in ('', 'nan'):
                    pts[(r['dataset'], m, r['codec'])].append((float(r['bpp']), float(r[m]), int(r['n_seq'])))
    out = {}
    for (ds, m, lab), v in pts.items():
        better = (lambda q, best: q < best) if m == 'lpips' else (lambda q, best: q > best)
        env, best = [], None  # upper envelope: drop points that a cheaper point already beats
        for x, y, _ in sorted(v):
            if best is None or better(y, best):
                env.append((x, y)); best = y
        out[(ds, m, lab)] = (env, v[0][2])
    return out


def rd_figure(a, spec, title, path, skip):
    c = curves_csv(a, title, skip)
    labs = [l for l in spec if l not in skip and any(k[2] == l for k in c)]
    fig, axes = plt.subplots(4, 4, figsize=(7.16, 7.0))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.925, bottom=0.05, wspace=0.27, hspace=0.36)
    for i, ds in enumerate(DATASETS):
        for j, (m, ylab) in enumerate(METRICS):
            ax = axes[i, j]
            n = 0
            # x range: the anchor's rate range widened by --xpad on each side; points outside are not drawn
            vtm = [x for x, _ in c.get((ds, m, 'VTM'), ([], 0))[0]]
            lo, hi = (min(vtm) / a.xpad, max(vtm) * a.xpad) if vtm else (1e-4, 10)
            xs, ys = [], []
            for lab in labs:
                if (ds, m, lab) not in c or not c[(ds, m, lab)][0]:
                    continue
                xy, n = c[(ds, m, lab)]
                xy = [(x, y) for x, y in xy if lo <= x <= hi]
                if not xy:
                    continue
                xs += [x for x, _ in xy]; ys += [y for _, y in xy]
                col, mk = STYLE[lab]
                ax.plot(*zip(*xy), LINE[spec[lab][2]], color=col, marker=mk, ms=3.8, lw=1.1, mew=0.4, mec='k', zorder=3)
            ax.set_xscale('log')
            if xs:  # axes tight around the drawn points
                ax.set_xlim(min(xs) / 1.25, max(xs) * 1.25)
                pad = 0.06 * (max(ys) - min(ys) or 1)
                ax.set_ylim(min(ys) - pad, max(ys) + pad)
            if m == 'lpips':
                ax.invert_yaxis()
            ax.set_facecolor('#e8eef5')  # light blue-grey panel, white grid
            ax.grid(True, which='major', color='white', lw=0.8)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(labelsize=6, length=2, pad=1.5, color='#7a8a9c')
            ax.text(0.03, 0.97, ylab, transform=ax.transAxes, ha='left', va='top', fontsize=6.8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#e8eef5', ec='none', alpha=0.85), zorder=4)
            if i == 3:
                ax.set_xlabel('bpp', fontsize=7, labelpad=1)
        # dataset name centred above its row
        l, r = axes[i, 0].get_position(), axes[i, 3].get_position()
        fig.text((l.x0 + r.x1) / 2, l.y1 + 0.006, ds, ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    handles = [Line2D([], [], ls=LINE[spec[l][2]], color=STYLE[l][0], marker=STYLE[l][1], ms=4, lw=1.2, label=l)
               for l in labs]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 6), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.2)
    fig.savefig(path); plt.close(fig)
    print(f'wrote {path}')
    for ds in DATASETS:
        print(f'  {title} {ds}: ' + ', '.join(f'{l} n={c[(ds, "psnr_y", l)][1]}' for l in labs if (ds, 'psnr_y', l) in c))


def table_bd(tex):
    """{(preset, row label): {(dataset, metric): BD-rate}} from the LD / RA tables (cells without a number left out)."""
    out, preset = {}, None
    cols = [(d, m) for d in DATASETS for m in ('PSNR', 'MS-SSIM', 'LPIPS', 'VMAF')]
    for line in tex.read_text().splitlines():
        if 'label{tab:bd-rate-ld}' in line: preset = 'LD'
        if 'label{tab:bd-rate-ra}' in line: preset = 'RA'
        if preset and line.count('&') == 16 and line.rstrip().endswith('\\\\') and 'Codec' not in line:
            cells = line.rstrip(' \\').split('&')
            lab = re.sub(r'\\revision\{([^}]*)\}', r'\1', cells[0])
            lab = re.sub(r'~?\\cite\{[^}]*\}|\\revision|[{}]', '', lab).strip()
            vals = [re.sub(r'\\revision\{([^}]*)\}', r'\1', c).strip() for c in cells[1:]]
            out[(preset, lab)] = {k: float(v) for k, v in zip(cols, vals) if re.fullmatch(r'-?[\d.]+', v)}
    return out


# Radar axes as in the original figure: encoding / decoding FPS and PSNR, MS-SSIM, VMAF BD-rate on HEVC B-E and UVG
# clockwise from the top: UVG quality axes on the right, the two FPS axes side by side at the bottom, HEVC on the left
RADAR_AXES = [('bd', ('UVG', 'PSNR'), 'UVG\nPSNR', 20), ('bd', ('UVG', 'MS-SSIM'), 'UVG\nMS-SSIM', 60),
              ('bd', ('UVG', 'VMAF'), 'UVG\nVMAF', 100),
              ('fps', 'dec_fps', 'Dec FPS', 155), ('fps', 'enc_fps', 'Enc FPS', 205),
              ('bd', ('HEVC B-E', 'VMAF'), 'HEVC\nVMAF', 260), ('bd', ('HEVC B-E', 'MS-SSIM'), 'HEVC\nMS-SSIM', 300),
              ('bd', ('HEVC B-E', 'PSNR'), 'HEVC\nPSNR', 340)]
# background / label colour of the rate-distortion sector (top) and the complexity sector (bottom)
SECTOR = {'bd': ('#f0f4fa', '#2b5d8c', 'Rate\u2013distortion (BD-rate)', 0, 250),
          'fps': ('#fdf1e4', '#b35900', 'Complexity (throughput)', 180, 110)}
RADAR = {'LD': ['DCVC-FM', 'DCVC-RT', 'DCVC-UF (LD)', 'PNVC', 'MaskCRT'],
         'RA': ['PNVC', 'NVRC', 'GIViC', 'DCVC-UF (HT-S)', 'BRHVC']}
RADAR_STYLE = {'DCVC-FM': '#E5487D', 'DCVC-RT': '#E4570F', 'PNVC': '#9B95C9', 'MaskCRT': '#F7B32B',
               'NVRC': '#7F7F7F', 'GIViC': '#F7B32B', 'DCVC-UF (LD)': '#4FB0A0', 'DCVC-UF (HT-S)': '#2E7D6B',
               'BRHVC': '#5BC0EB'}
R0 = 0.1  # radius of the worst value on an axis


def radar_figure(a, path, skip):
    """Radar plot in the style of the original Fig. 1. Each axis runs from the worst to the best codec of the panel
    (radius R0 .. 1, FPS on a log scale), with the actual values printed on three rings."""
    bd = table_bd(a.table)
    fps = {(r['preset'], r['codec']): r for r in csv.DictReader(a.complexity.open())}
    n = len(RADAR_AXES)
    ang = np.radians([d for *_, d in RADAR_AXES])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.7), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.10, right=0.90, top=0.79, bottom=0.07, wspace=0.40)
    handles = {}
    for ax, preset in zip(axes, ['LD', 'RA']):
        vals = {}
        for c in RADAR[preset]:
            r = fps.get((preset, c))
            row = bd.get((preset, r['table_label'])) if r else None
            v = [(float(r[k]) if r and r[k] else None) if kind == 'fps' else (row or {}).get(k) for kind, k, _, _ in RADAR_AXES]
            if c in skip or None in v:
                print(f'  radar {preset}: {c} left out (missing ' +
                      ', '.join(RADAR_AXES[i][2].replace(chr(10), ' ') for i, x in enumerate(v) if x is None) + ')')
                continue
            vals[c] = v
        # per axis: score -> radius; FPS higher is better (log), BD-rate lower is better
        score = lambda i, x: math.log10(x) if RADAR_AXES[i][0] == 'fps' else -x
        lo = [min(score(i, v[i]) for v in vals.values()) for i in range(n)]
        hi = [max(score(i, v[i]) for v in vals.values()) for i in range(n)]
        rad = lambda i, x: R0 + (1 - R0) * ((score(i, x) - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 1.0)
        for c, v in vals.items():
            rr = [rad(i, x) for i, x in enumerate(v)]
            ax.plot(np.r_[ang, ang[:1]], rr + rr[:1], color=RADAR_STYLE[c], lw=2.0, zorder=3)
            ax.fill(np.r_[ang, ang[:1]], rr + rr[:1], color=RADAR_STYLE[c], alpha=0.05, zorder=2)
            handles[c] = plt.Line2D([], [], color=RADAR_STYLE[c], lw=2.0, marker='o', ms=6, label=c)
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_thetalim(0, 2*np.pi)
        ax.set_ylim(0, 1.0)
        for col, _, lab, mid, width in SECTOR.values():  # tinted sectors separating quality and complexity axes
            ax.bar(np.radians(mid), 1.0, width=np.radians(width), bottom=0, color=col, zorder=0)
        ax.set_yticks([R0 + (1 - R0)*f for f in (0, 0.25, 0.58, 0.9)]); ax.set_yticklabels([])
        ax.grid(color='0.8', lw=0.6)
        ax.spines['polar'].set_color('black'); ax.spines['polar'].set_linewidth(1.0)
        for i, t in enumerate(ang):  # values on three rings, rotated along the axis
            rot = ((90 - np.degrees(t)) % 180) - 90
            for f in (0.25, 0.58, 0.9):
                sc = lo[i] + f * (hi[i] - lo[i])
                x = 10**sc if RADAR_AXES[i][0] == 'fps' else -sc
                txt = f'{x:.3g}' if RADAR_AXES[i][0] == 'fps' else f'{x:.1f}'
                ax.text(t, R0 + (1 - R0)*f, txt, fontsize=9, color='0.5', ha='center', va='center', rotation=rot,
                        rotation_mode='anchor', zorder=4)
        ax.set_xticks(ang); ax.set_xticklabels([])
        for t, (kind, _, lab, _) in zip(ang, RADAR_AXES):  # short horizontal axis names just outside the rim
            rot = (-np.degrees(t) + 90) % 180 - 90  # tangent to the circle, kept upright
            ax.text(t, 1.21, lab, fontsize=12.5, linespacing=0.95, color=SECTOR[kind][1], ha='center', va='center',
                    rotation=rot, rotation_mode='anchor')
        ax.set_title('Low Delay' if preset == 'LD' else 'Random Access', fontsize=14, fontweight='bold', pad=38)
    fig.legend(handles=list(handles.values()), loc='upper center', ncol=len(handles), fontsize=11.5, frameon=False,
               handlelength=1.8, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path); plt.close(fig)
    print(f'wrote {path}')

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--table', type=Path, default=REPO/'tables/table_ra.tex')
    p.add_argument('--rd-csv', type=Path, default=REPO/'data/rd/rd_curves.csv', help='from scripts/build_rd_csv.py')
    p.add_argument('--complexity', type=Path, default=REPO/'figures/data/rd_complexity.csv')
    p.add_argument('--out', type=Path, default=REPO/'figures')
    p.add_argument('--xpad', type=float, default=4.0, help='x range = VTM bpp range widened by this factor')
    p.add_argument('--skip', nargs='*', default=['AVM'], help='codecs left out (AVM: wrong encoder configuration)')
    a = p.parse_args()
    plt.rcParams.update({'font.family': 'serif', 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
    rd_figure(a, LD_ALL, 'LD', a.out/'rd_lowdelay_v2.pdf', a.skip)
    rd_figure(a, RA_ALL, 'RA', a.out/'rd_randomaccess_v2.pdf', a.skip)
    with plt.rc_context({'font.family': 'sans-serif'}):  # the original radar figure uses the default sans font
        radar_figure(a, a.out/'radar_v2.pdf', a.skip)


if __name__ == '__main__':
    main()
