"""Plot only x265, VTM and the supplied four-point CC00 WD dataset averages."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/rd-matplotlib')
from pathlib import Path
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/'results/rd_dataset_average'
parser=argparse.ArgumentParser()
parser.add_argument('--cc00-rate-unit', choices=['bytes','bits'], required=True, help='Unit of supplied CC00 numbers; provenance must be confirmed separately')
a=parser.parse_args()
out=root/('x265_vtm_cc00_bits_assumed' if a.cc00_rate_unit=='bytes' else 'x265_vtm_cc00_bits_declared');out.mkdir(exist_ok=True)
methods=['x265','VTM','Cool-Chic WD (CC00)']
rows=[r for r in csv.DictReader((root/'dataset_rate_averages.csv').open()) if r['method'] in methods]
assert len(rows)==24
for r in rows:
 if r['method']=='Cool-Chic WD (CC00)' and a.cc00_rate_unit=='bytes':
  r['bpp']=format(float(r['bpp'])*8,'.9g')
with (out/'dataset_rate_averages.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
styles={'x265':('#555555','s'),'VTM':('#d14936','^'),'Cool-Chic WD (CC00)':('#187bbf','o')}
metrics={'psnr':'RGB PSNR (dB) ↑','vmaf':'VMAF ↑','lpips':'LPIPS ↓','dists':'DISTS ↓'}
plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'savefig.dpi':180})
for dataset in ['HEVC-B','UVG']:
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for ax,(metric,label) in zip(axes.flat,metrics.items()):
  for method in methods:
   rs=sorted([r for r in rows if r['dataset']==dataset and r['method']==method],key=lambda r:float(r['bpp']))
   assert len(rs)==4 and len({r['lambda'] for r in rs})==4
   color,marker=styles[method]
   ax.plot([float(r['bpp']) for r in rs],[float(r[metric]) for r in rs],color=color,marker=marker,markersize=7,linewidth=2,label=method)
  ax.set_xscale('log');ax.set_xlabel('Average bits/pixel/frame (log scale)');ax.set_ylabel(label);ax.grid(True,which='both',alpha=.22)
 fig.suptitle(f'{dataset} — Dataset-average RD curves',fontsize=17,y=.98)
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
 note=('CC00 rate ×8, assuming supplied values are bytes/pixel; table provenance unconfirmed.' if a.cc00_rate_unit=='bytes' else 'CC00 supplied rates declared bits/pixel; no conversion applied.')
 fig.text(.5,.025,note+'\nLPIPS: codec AlexNet; CC00 unspecified. Metric-pipeline equivalence not verified.',ha='center',fontsize=9,color='#444444')
 fig.tight_layout(rect=[0,.075,1,.9])
 for ext in ['png','pdf']:fig.savefig(out/f'{dataset}_RD.{ext}')
 plt.close(fig)
(out/'README.md').write_text('Only x265, VTM and the eight user-supplied CC00 WD dataset-average rows are included. Each method has four points per dataset. No pure WD or SWD rows from the long per-video table are used.\n\nCSV schema matches the requested schema. QP labels in lambda identify conventional codec rate points, not neural optimization lambdas. PSNR for local codecs is the mean frame PSNR, then equal-sequence dataset average. CC00 quality aggregates are used as supplied; rates are converted according to the explicit CLI unit.\n\nMS-SSIM is absent from these comparison figures because it was not measured for x265/VTM; available CC00 values are retained in the CSV. Local LPIPS uses AlexNet; CC00 backbone is unspecified. VMAF and other pipeline equivalence has not been independently verified.\n')
with (out/'README.md').open('a') as f:
 f.write('\nInput CC00 rate unit selected: '+a.cc00_rate_unit+'. The bytes selection multiplies only CC00 rates by 8; its provenance remains unconfirmed. Original data are not overwritten.\n')
print(out)
