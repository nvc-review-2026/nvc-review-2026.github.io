#!/usr/bin/env python3
"""Reproduce equal-sequence dataset RD summaries from codec results and pasted tables."""
import csv
import io
import json
import math
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rd-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rd_dataset_average'
FIELDS = ['dataset','method','lambda','bpp','psnr','msssim','vmaf','lpips','dists']
METRICS = FIELDS[3:]
VIDEOS = {'HEVC-B': {'BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene'},
          'UVG': {'Beauty','Bosphorus','HoneyBee','Jockey','ReadySetGo','ShakeNDry','YachtRide'}}
LABELS = {'psnr':'RGB PSNR (dB) ↑','msssim':'MS-SSIM ↑','vmaf':'VMAF ↑',
          'lpips':'LPIPS-Alex ↓','dists':'DISTS ↓'}
METHOD_LABELS = {
 'SWD + pixel consistency (source anchor), d=1, w0.2':'SWD + pixel consistency (source anchor)',
 'WD + pixel consistency (source anchor), d=1, w0.2':'WD + pixel consistency (source anchor)',
 'Cool-Chic WD (CC00)':'Cool-Chic WD (CC00)',
}

def write_csv(name, fields, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.9g}' if isinstance(v,float) else v) for k,v in r.items() if k in fields})

def dataset(video):
    return next(d for d,vs in VIDEOS.items() if video in vs)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    supplied=(OUT/'user_supplied_tables.txt').read_text()
    lines=supplied[supplied.index('video\trun\t'):].splitlines()
    header=lines[0].split('\t')
    pasted=[]
    for line in lines[1:]:
        if '\tvideo_' not in line: continue
        r=dict(zip(header,line.split('\t')))
        # Keep original variant labels and the six requested numerical metrics.
        row=dict(dataset=dataset(r['video']),video=r['video'],method=r['method'],
                 **{'lambda':format(float(r['lambda']),'.6g')},run=r['run'],
                 metric_pipeline=r['metric_pipeline'],lpips_backbone='alex',source='user per-video table')
        for dest,src in dict(bpp='bpp',psnr='psnr',msssim='ms_ssim',vmaf='vmaf',lpips='lpips_alex',dists='dists').items():
            row[dest]=float(r[src])
        pasted.append(row)
    assert len(pasted)==229, f'Unexpected input count: {len(pasted)}'
    write_csv('supplied_per_sequence.csv',FIELDS+['video','run','metric_pipeline','lpips_backbone','source'],pasted)
    baseline=[]
    for line in supplied.splitlines():
        if line.startswith(('HEVC-B,Cool-Chic WD (CC00),','UVG,Cool-Chic WD (CC00),')):
            # Final data row is followed by Chinese prose in the original message.
            cells=next(csv.reader([line])); cells[8]=cells[8].split()[0]
            row=dict(zip(FIELDS,cells[:9]))
            for k in METRICS: row[k]=float(row[k])
            row['lambda']=format(float(row['lambda']),'.6g')
            row.update(n_sequences=len(VIDEOS[row['dataset']]),coverage='provided dataset average; not independently verified',
                       videos=';'.join(sorted(VIDEOS[row['dataset']])),metric_pipeline='user aggregate; unspecified',source='user dataset averages',full_dataset=True)
            baseline.append(row)
    assert len(baseline)==8
    codec_rows=list(csv.DictReader((ROOT/'results/x265_vtm_rgb_96_gop32/results.csv').open()))
    assert len(codec_rows)==96
    per_sequence=list(pasted)
    for r in codec_rows:
        video=r['sequence'].split('_1920x1080')[0]
        assert video in VIDEOS[r['dataset']] and int(r['frames'])==96
        bpp=float(r['bytes'])*8/(int(r['width'])*int(r['height'])*int(r['frames']))
        assert math.isclose(bpp,float(r['bpp']),rel_tol=1e-12)
        per_sequence.append(dict(dataset=r['dataset'],video=video,method={'x265':'x265','vtm':'VTM'}[r['codec']],
            **{'lambda':f"QP{r['qp']}"},bpp=bpp,psnr=float(r['psnr_rgb_frame_mean']),msssim=None,
            vmaf=float(r['vmaf_rgb_luma']),lpips=float(r['lpips']),dists=float(r['dists']),
            metric_pipeline='RGB full-resolution; BT.601 luma VMAF v0.6.1',source='local codecs',lpips_backbone='alex'))
    groups=defaultdict(list)
    for r in per_sequence:groups[(r['dataset'],r['method'],r['lambda'])].append(r)
    averages=[]
    for key,rs in sorted(groups.items()):
        vs={r['video'] for r in rs}
        assert len(vs)==len(rs),f'Duplicate sequence at {key}'
        d,m,rate=key
        row=dict(zip(FIELDS[:3],key))
        for k in METRICS:
            vals=[r[k] for r in rs if r[k] is not None]
            assert not vals or len(vals)==len(rs),f'Partial metric coverage {key} {k}'
            assert all(math.isfinite(x) for x in vals)
            row[k]=mean(vals) if vals else None
        row.update(n_sequences=len(vs),coverage=f'{len(vs)}/{len(VIDEOS[d])}',videos=';'.join(sorted(vs)),
                   metric_pipeline=';'.join(sorted({r['metric_pipeline'] for r in rs})),
                   source=rs[0]['source'],full_dataset=vs==VIDEOS[d])
        averages.append(row)
    averages+=baseline
    averages.sort(key=lambda r:(r['dataset'],r['method'],r['bpp']))
    full=[r for r in averages if r['full_dataset']]
    write_csv('dataset_rate_averages.csv',FIELDS,full)
    write_csv('dataset_rate_averages_all_available.csv',FIELDS,averages)
    write_csv('coverage.csv',FIELDS[:3]+['n_sequences','coverage','full_dataset','videos','metric_pipeline','source'],averages)
    write_csv('x265_vtm_dataset_rate_averages.csv',FIELDS,[r for r in averages if r['method'] in ['x265','VTM']])
    write_csv('per_sequence_normalized.csv',FIELDS+['video','metric_pipeline','source'],per_sequence)
    methods=sorted({r['method'] for r in averages})
    colors={m:plt.get_cmap('tab20')(i%20) for i,m in enumerate(methods)}
    colors.update({'x265':'#303030','VTM':'#bf2929','pure SWD':'#2488b5','Cool-Chic WD (CC00)':'#ae7c00'})
    markers=['o','s','^','D','v','P','X','<','>']
    def draw(ax, rows, metric, partial=False):
        for idx,m in enumerate(methods):
            rs=sorted([r for r in rows if r['method']==m and r[metric] is not None],key=lambda r:r['bpp'])
            if not rs:continue
            label=METHOD_LABELS.get(m,m)
            if partial:
                counts=sorted({r['n_sequences'] for r in rs}); label+=f" [n={','.join(map(str,counts))}]"
            if metric=='lpips' and m=='Cool-Chic WD (CC00)':label+=' (backbone unspecified)'
            ax.plot([r['bpp'] for r in rs],[r[metric] for r in rs],label=label,color=colors[m],
                    marker=markers[idx%len(markers)],markersize=5,linewidth=1.6,
                    linestyle='--' if m=='Cool-Chic WD (CC00)' else '-')
        ax.set_xscale('log');ax.set_xlabel('Mean bits / pixel / frame (log scale)')
        ax.set_ylabel(LABELS[metric]);ax.grid(True,which='both',alpha=.2)
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'pdf.fonttype':42})
    for d in VIDEOS:
        rows=[r for r in full if r['dataset']==d]
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        for ax,k in zip(axes.flat,LABELS):draw(ax,rows,k)
        axes[1,2].axis('off')
        handles,labels=axes[0,0].get_legend_handles_labels()
        axes[1,2].legend(handles,labels,loc='center',frameon=False,fontsize=9)
        axes[1,2].text(0,.05,'Equal sequence weights; full dataset coverage only.\nCC00: supplied aggregates; protocol not verified.\nCodec MS-SSIM unavailable; LPIPS uses AlexNet\nexcept CC00 (backbone unspecified).',transform=axes[1,2].transAxes,fontsize=9)
        fig.suptitle(f'{d}: dataset-average rate–distortion ({len(VIDEOS[d])} sequences)\nReported results; cross-pipeline comparability not yet verified',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.93])
        for ext in ['png','pdf']:fig.savefig(OUT/f'{d}_RD.{ext}')
        plt.close(fig)
        for k in LABELS:
            fig,ax=plt.subplots(figsize=(8.4,5.8));draw(ax,rows,k)
            ax.set_title(f'{d} — {LABELS[k]} (full dataset averages)')
            ax.legend(fontsize=7,framealpha=.9);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(OUT/f'{d}_{k}.{ext}')
            plt.close(fig)
    # Separate one-method-per-page plots keep sparse ablations from being mistaken
    # for full dataset averages, and avoid drawing lines across changing subsets.
    with PdfPages(OUT/'partial_coverage_exploratory.pdf') as pdf:
        for d,m in sorted({(r['dataset'],r['method']) for r in averages if not r['full_dataset']}):
            rows=[r for r in averages if r['dataset']==d and r['method']==m and not r['full_dataset']]
            subsets=sorted({r['videos'] for r in rows})
            fig,axes=plt.subplots(2,3,figsize=(14,8))
            for ax,k in zip(axes.flat,LABELS):
                for i,vs in enumerate(subsets):
                    rs=sorted([r for r in rows if r['videos']==vs and r[k] is not None],key=lambda r:r['bpp'])
                    if rs:ax.plot([r['bpp'] for r in rs],[r[k] for r in rs],marker='o',label=f'Subset {i+1}: n={rs[0]["n_sequences"]}')
                ax.set_xscale('log');ax.set_xlabel('Mean bpp');ax.set_ylabel(LABELS[k]);ax.grid(alpha=.2);ax.legend(fontsize=8)
            axes[1,2].axis('off')
            import textwrap
            note='PARTIAL COVERAGE — not full dataset performance.\n\n'+'\n\n'.join(f'Subset {i+1}: '+', '.join(vs.split(';')) for i,vs in enumerate(subsets))
            axes[1,2].text(0,1,'\n'.join(textwrap.fill(t,48) for t in note.split('\n')),va='top',fontsize=10)
            fig.suptitle(f'{d} — {m}',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)
    print('Parsed supplied per-sequence rows:',len(pasted))
    print('All dataset/method/rate groups:',len(averages),'Full-coverage groups:',len(full))
    print('Outputs:',OUT)

if __name__=='__main__':main()
