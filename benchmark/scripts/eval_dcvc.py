#!/usr/bin/env python3
"""DCVC-DC / DCVC-FM on the CTC YUV sequences: run each model's own test_video.py, then the same metrics as the
CTC codecs (eval_ctc_chunks.metrics: PSNR, MS-SSIM, VMAF, VMAF NEG, LPIPS-Alex) on the decoded YUV (resumable).

Run inside `conda activate compression` (metrics); the models run with --dcvc-python (the dcvc env).
Sequences: dataset_config.json of the Isambard toolkit, 8-bit only (the DCVC YUV readers are 8-bit, so the 10-bit
AOM-CTC sequences are skipped). All frames, one I frame (intra period 9999 for DCVC-FM, as -1 codes every frame intra there; DCVC-DC gets
gop=9999 since its loop has no -1), 4 rates with each model's default q indexes, YUV420 models, one GPU worker.
bpp is the entropy-model estimate for both (write_stream=0): the local DCVC-FM bitstream decoder does not reproduce the
encoder (decoded PSNR ~19 dB), so the reconstruction is taken from the encoder pass. DCVC-FM runs from a copy
($NVC_ROOT/DCVC/DCVC-FM-eval) whose fast path is patched to write that reconstruction.
Per sequence the decoded YUVs go to --tmp and are deleted after the metrics; results are cached as
<out>/<model>/<class>/<sequence>/rate_<i>.json. Writes <out>/dcvc_metrics_sequences.csv and dcvc_metrics_datasets.csv.
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_ctc_chunks import METRICS, dataset_of, metrics  # noqa: E402

DCVC = Path(os.environ.get('DCVC_ROOT', 'DCVC'))
MODELS = {
    'dcvc-fm': dict(dir=Path(NVC_ROOT, 'DCVC/DCVC-FM-eval'),  # copy with the enc/dec assert relaxed
                    args=['--model_path_i', DCVC/'checkpoints/cvpr2024_image.pth.tar',
                          '--model_path_p', DCVC/'checkpoints/cvpr2024_video.pth.tar',
                          '--write_stream', '0', '--save_decoded_frame', '1', '--cuda', '1', '--worker', '1']),
    'dcvc-dc': dict(dir=DCVC/'DCVC-family/DCVC-DC',
                    args=['--i_frame_model_path', f'{NVC_ROOT}/DCVC/checkpoints/DCVC-DC/cvpr2023_image_yuv420_psnr.pth.tar',
                          '--p_frame_model_path', f'{NVC_ROOT}/DCVC/checkpoints/DCVC-DC/cvpr2023_video_yuv420_psnr.pth.tar',
                          '--yuv420', '1', '--write_stream', '0', '--save_decoded_frame', '1', '--cuda', '1', '--worker', '1']),
}


def run_model(a, model, cls, base, seq, info, tmp):
    """Encode/decode one sequence at 4 rates; returns {rate_idx: (bpp, recon_path)}."""
    m = MODELS[model]
    ent = dict(width=info['width'], height=info['height'], frames=info['frames'])
    # one I frame: DCVC-FM tests frame_idx % intra_period == 0, so -1 would make every frame intra
    ent.update(intra_period=9999) if model == 'dcvc-fm' else ent.update(gop=9999)
    cfg = dict(root_path=str(a.source), test_classes={cls: dict(test=1, base_path=base, src_type='yuv420', sequences={seq: ent})})
    (tmp/'cfg.json').write_text(json.dumps(cfg))
    extra = (['--stream_path', tmp/'bin'] if model == 'dcvc-fm' else ['--decoded_frame_path', tmp/'rec', '--stream_path', tmp/'bin'])
    with open(tmp/'model.log', 'w') as log:
        subprocess.run([a.dcvc_python, 'test_video.py', *m['args'], *extra, '--rate_num', '4',
                        '--test_config', tmp/'cfg.json', '--output_path', tmp/'out.json'],
                       cwd=m['dir'], stdout=log, stderr=subprocess.STDOUT, check=True)
    res = json.loads((tmp/'out.json').read_text())[cls][seq]
    out = {}
    for k, r in res.items():
        i = int(k)
        if model == 'dcvc-fm':
            rec = tmp/'bin'/cls/f"{seq}_q{r.get('q_index_i', '')}.yuv"
            rec = rec if rec.exists() else Path(sorted(glob.glob(str(tmp/'bin'/cls/f'{seq}_q*.yuv')))[i])
        else:
            rec = Path(glob.glob(str(tmp/'rec'/seq/f'{i}_*'/'out.yuv'))[0])
        out[i] = (r['ave_all_frame_bpp'], rec)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=['dcvc-fm', 'dcvc-dc'], choices=list(MODELS))
    p.add_argument('--toolkit', type=Path, default=Path(NVC_ROOT, 'codecs/codec_results'))
    p.add_argument('--source', type=Path, default=Path(NVC_ROOT, 'codecs/source_yuv'))
    p.add_argument('--out', type=Path, default=Path('results/dcvc_metrics'))
    p.add_argument('--tmp', type=Path, default=Path(NVC_ROOT, 'DCVC/tmp'))
    p.add_argument('--classes', nargs='+', help='default: all 8-bit classes')
    p.add_argument('--sequence', help='only this sequence (substring), for testing')
    p.add_argument('--dcvc-python', default=str(Path.home()/'miniforge3/envs/dcvc/bin/python'))
    p.add_argument('--matrix', default='bt709')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--ffmpeg', type=Path, default=Path.home()/'.local/ffmpeg-nvenc/bin/ffmpeg')
    p.add_argument('--vmaf-ffmpeg', type=Path, default=Path.home()/'opt/ffmpeg-vmaf/bin/ffmpeg')
    a = p.parse_args()
    os.environ['LD_LIBRARY_PATH'] = f"{a.ffmpeg.parent.parent/'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    cfg = json.loads((a.toolkit/'dataset_config.json').read_text())['test_classes']
    depth = {(r['sequence_class'], r['sequence']): int(r['bitdepth']) for r in csv.DictReader((a.toolkit/'manifest.csv').open())}
    jobs = [(cls, v['base_path'], s, info) for cls, v in cfg.items() if not a.classes or cls in a.classes
            for s, info in v['sequences'].items()
            if depth.get((cls, s[:-4]), 8) == 8 and (not a.sequence or a.sequence in s)]
    print(f'{len(jobs)} sequences x {len(a.models)} models', flush=True)

    import lpips, torch
    alex = lpips.LPIPS(net='alex', verbose=False).eval().to(a.device)
    for model in a.models:
        for n, (cls, base, seq, info) in enumerate(jobs, 1):
            d = a.out/model/cls/seq[:-4]
            if all((d/f'rate_{i}.json').exists() for i in range(4)):
                continue
            tmp = a.tmp/model; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
            try:
                pts = run_model(a, model, cls, base, seq, info, tmp)
                w, h, f = info['width'], info['height'], info['frames']
                for i, (bpp, rec) in sorted(pts.items()):
                    assert rec.stat().st_size == w*h*3//2*f, f'{rec}: size {rec.stat().st_size} != {w*h*3//2*f}'
                    fr = metrics(a, a.source/base/seq, rec, w, h, 8, 0, f, alex, tmp)
                    row = dict(model=model, cls=cls, sequence=seq[:-4], rate=f'rate_{i}', width=w, height=h, frames=f, bpp=bpp,
                               **{k: sum(x[k] for x in fr)/f for k in METRICS})
                    d.mkdir(parents=True, exist_ok=True); (d/f'rate_{i}.json').write_text(json.dumps(row, indent=1))
                    print(f"[{n}/{len(jobs)}] {model} {cls} {seq[:-4]} rate_{i}  bpp {bpp:.4f}  PSNR-Y {row['psnr_y']:.2f}  "
                          f"VMAF {row['vmaf']:.1f}  LPIPS {row['lpips_alex']:.4f}", flush=True)
            except Exception as e:
                print(f'[{n}/{len(jobs)}] FAIL {model} {cls} {seq}: {e} (log: {tmp}/model.log)', flush=True)
                continue
            shutil.rmtree(tmp, ignore_errors=True)

    rows = [json.loads(f.read_text()) for f in sorted(a.out.glob('*/*/*/rate_*.json'))]
    if not rows:
        return
    with (a.out/'dcvc_metrics_sequences.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    g = defaultdict(list)
    for r in rows:
        g[(r['model'], dataset_of(r['cls']), r['rate'])].append(r)
    with (a.out/'dcvc_metrics_datasets.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'dataset', 'rate', 'sequences', 'bpp', *METRICS])
        for (m, ds, rt), v in sorted(g.items()):
            w.writerow([m, ds, rt, len(v), *(sum(r[k] for r in v)/len(v) for k in ['bpp', *METRICS])])
    print(f"wrote {a.out/'dcvc_metrics_sequences.csv'} and dcvc_metrics_datasets.csv", flush=True)


if __name__ == '__main__':
    main()
