#!/usr/bin/env python3
"""Short local profiling of the INR codecs HiNeRV and NVRC on 1080p (run alone, nothing else on the GPU or CPU).

INR encoding = overfitting a network, so nothing is trained to completion. Each rate point trains a few epochs of
every stage, the per-epoch time of each stage is measured from the log (difference of the last "Train - Epoch k"
line of consecutive epochs; epoch 0 of each stage carries torch.compile/setup and is dropped when there are >= 2),
and the encoding time is extrapolated with the original epoch schedule:
    enc_seconds = sum_stage full_epochs_stage * mean_epoch_seconds_stage   (evaluations / entropy coding excluded)
    HiNeRV: main 300 + prune 60 + QAT 30 (cfgs/train/hinerv_1920x1080.txt);  NVRC: s1 360 + s2 30 (lamb 1.0)
Decoding is real, on the short-trained model: load bitstream -> entropy decode -> inference of every frame
(HiNeRV: hinerv_main.py --bitstream --eval-only --bitstream-q 6 without torch.compile; NVRC: main_nvrc.py
--eval-only resumed from the s2 run, window from "Decoding model weights" to "Evaluation completed").
Memory: scripts/peakmon.py (host RSS of the process tree, GPU memory.used above idle). Encoding peak = training
processes; decoding peak = the decode process (NVRC: from the "Decoding" mark on).

Sequence MCL-JCV videoSRC01, 150 frames 1080p. Modes as the GH200 runs: full = one model for 150 frames;
seg = GoP of 30 frames (5 segments), only segment 0 is run and the time is scaled x5 (enc_fps is the same).
Seg models scale the temporal grid size by 30/150 (HiNeRV base grid T 150 -> 30, NVRC 50 -> 10).
The exact GH200 per-rate model configs are not in the repos, so decoder widths are scaled to reproduce the GH200
kMACs/px of each codec/mode/rate (data/profiling/inr_rd_runs.csv); compare kmacs_px / params with that file.

Writes results/complexity/inr_gpu_bench.csv (resumable: rows with status ok are skipped) and prints
mean +- std over the rate points per codec x mode.
  python scripts/bench_inr_gpu.py                 # everything (2 codecs x 2 modes x 5 rates)
  python scripts/bench_inr_gpu.py --quick         # HiNeRV seg s, 1 epoch per stage (smoke test, not saved)
"""
import argparse, csv, os, re, statistics, sys, time
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import peakmon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT/'DVC-Review-PIEEE/submodules'
HINERV, NVRC = SUB/'hinerv', SUB/'nvrc'
CONDA = Path.home()/'miniforge3/envs'
DATA = Path(NVC_ROOT, 'inr_bench/png')          # videoSRC01/0001..0150.png, videoSRC01-seg0000/ (1..30)
RUNS = Path(NVC_ROOT, 'inr_bench/runs')
CSV = ROOT/'results/complexity/inr_gpu_bench.csv'
SEQ, FRAMES, SEG_FRAMES, W, H = 'videoSRC01', 150, 30, 1920, 1080
RATES = {'HiNeRV': ['s', 'm', 'l', 'xl', 'xxl'], 'NVRC': ['xs', 's', 'm', 'l', 'xl']}
SCHEDULE = {'HiNeRV': {'main': 300, 'prune': 60, 'qat': 30}, 'NVRC': {'s1': 360, 's2': 30}}
FIELDS = ['codec', 'mode', 'rate', 'status', 'gpu', 'frames', 'segments', 'params_M', 'params_seq_M', 'kmacs_px', 'gh200_kmacs_px',
          'measured_epochs', 'epoch_s', 'enc_s_segment', 'enc_s_sequence', 'enc_fps', 'dec_s', 'dec_fps',
          'host_enc_mib', 'gpu_enc_mib', 'host_dec_mib', 'gpu_dec_mib', 'note']
EPOCH = re.compile(r'Train(?: \((\w+)\))? - Epoch (\d+) \[')
ENV = {**os.environ, 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': '0', 'TOKENIZERS_PARALLELISM': 'false'}


def launch(env, dynamo):
    return [CONDA/env/'bin/python', '-m', 'accelerate.commands.launch', '--num_processes', '1', '--num_machines', '1',
            '--mixed_precision', 'fp16', '--dynamo_backend', dynamo]


def run(cmd, cwd, log):
    print('  $', ' '.join(map(str, cmd))[:300], flush=True)
    try:
        r = peakmon.run(cmd, gpu=0, cwd=cwd, env=ENV)
    except peakmon.subprocess.CalledProcessError as e:
        log.write_text(str(e.output)); raise RuntimeError(f'{cmd[4:6]} failed, see {log}:\n{e.output}') from None
    log.write_text('\n'.join(f'{t:9.2f} {l}' for t, l in r['lines']))
    return r


def mark(r, pat, last=False):
    hits = [t for t, l in r['lines'] if re.search(pat, l)]
    if not hits:
        raise RuntimeError(f'no log line matching {pat!r}')
    return hits[-1] if last else hits[0]


def epoch_times(r, starts):
    """starts: {stage_label: regex of the line just before the stage's first epoch}; label None = unlabelled."""
    ends = {}
    for t, l in r['lines']:
        m = EPOCH.search(l)
        if m:
            ends.setdefault(m.group(1), {})[int(m.group(2))] = t
    out = {}
    for label, pat in starts.items():
        e = ends[label]
        ts = [mark(r, pat)] + [e[k] for k in sorted(e)]
        d = [b - a for a, b in zip(ts, ts[1:])]
        out[label] = (statistics.mean(d[1:]) if len(d) > 1 else d[0], len(d))
    return out


def num(r, pat):
    m = [re.search(pat, l) for _, l in r['lines']]
    m = [x for x in m if x]
    return float(m[0].group(1)) if m else None


GH200_RD = ROOT/'DVC-Review-PIEEE/data/profiling/inr_rd_runs.csv'


def target_kmacs(codec, mode, rate):
    for r in csv.DictReader(open(GH200_RD)):
        if r['codec'] == codec and r['mode'] == mode and r['name'] == SEQ and r['rate'].split('-lamb')[0] == rate:
            return float(r['kmacs_per_px'])
    return None


# ---------------------------------------------------------------- HiNeRV
def hinerv(mode, rate, ep, wd):
    # The GH200 rate configs are not in the repo: start from uvg-hinerv-s (280 ch, 87.7 kMACs/px measured) and
    # scale the decoder width so kMACs/px matches the GH200 value of this codec/mode/rate (MACs ~ channels^2).
    cfg = (HINERV/'cfgs/models/uvg-hinerv-s_1920x1080.txt').read_text().split()
    i = cfg.index('--channels'); cfg[i + 1] = str(max(16, round(280 * (target_kmacs('HiNeRV', mode, rate) / 87.7) ** .5 / 16) * 16))
    if mode == 'seg':
        i = cfg.index('--base-grid-size'); cfg[i + 1] = str(int(cfg[i + 1]) * SEG_FRAMES // FRAMES)
    name = SEQ if mode == 'full' else f'{SEQ}-seg0000'
    base = ['hinerv_main.py', '--dataset', DATA, '--dataset-name', name, '--output', wd,
            *(HINERV/'cfgs/train/hinerv_1920x1080.txt').read_text().split(), *cfg,
            '--batch-size', 144, '--eval-batch-size', 1, '--grad-accum', 1, '--log-eval', 'false', '--seed', 0]
    enc = run([*launch('hinerv', 'inductor'), *base, '--epochs', ep, '--prune-epochs', ep, '--quant-epochs', ep],
              HINERV, wd/'enc.log')
    out = [l.split('Output dir: ')[1].strip() for _, l in enc['lines'] if 'Output dir: ' in l][0]
    et = epoch_times(enc, {None: r'Start main training', 'Pruning': r'Start pruning fine-tuning',
                           'QAT': r'Start quant fine-tuning'})
    dec = run([*launch('hinerv', 'no'), *base, '--bitstream', out, '--bitstream-q', 6, '--eval-only'],
              HINERV, wd/'dec.log')
    t0 = mark(dec, r'Restore model weights from bitstream')
    t1 = mark(dec, rf'Eval - \[(\d+)/\1\]', last=True)
    macs_g = num(enc, r'MACs: ([\d.]+)G')
    return dict(params_M=num(enc, r'^\s*All: ([\d.]+)M') or num(enc, r'All: ([\d.]+)M'),
                kmacs_px=macs_g * 1e6 / (W * H) if macs_g else None,
                epoch_s={'main': et[None], 'prune': et['Pruning'], 'qat': et['QAT']},
                dec_s=t1 - t0, host_enc=enc['host_mib'], gpu_enc=enc['gpu_mib'],
                host_dec=dec['host_mib'], gpu_dec=dec['gpu_mib'])


# ---------------------------------------------------------------- NVRC
def nvrc_model_cfg(mode, rate, wd):
    # GH200 configs are not in the repo: start from mcl-jcv m (decoder [512,256,128,64], 382.4 kMACs/px measured) and
    # scale the decoder widths to the GH200 kMACs/px of this mode/rate (MACs ~ channels^2), T grid 50 per 150 frames.
    import yaml
    cfg = yaml.safe_load((NVRC/'scripts/configs/nvrc/models/mcl-jcv_hinerv-v2_m_1920x1080.yaml').read_text())
    k = target_kmacs('NVRC', mode, rate) or 4.4 * target_kmacs('NVRC', 'seg', rate)  # full xs/s not in GH200 csv
    c = max(16, round(512 * (k / 382.4) ** .5 / 16) * 16)
    cfg['config']['decoder']['channels'] = [c, c // 2, c // 4, c // 8]
    g = cfg['config']['base_encoding']['base_grid_size']
    g[0] = 50 * (SEG_FRAMES if mode == 'seg' else FRAMES) // FRAMES
    p = wd/'model.yaml'; p.write_text(yaml.safe_dump(cfg)); return p


def nvrc(mode, rate, ep, wd):
    c = NVRC/'scripts/configs'
    nf = ['--num-frames', SEG_FRAMES, '--intra-period', SEG_FRAMES] if mode == 'seg' else []
    common = ['main_nvrc.py', '--output', wd, '--train-task-config', c/'tasks/overfit/l1_ms-ssim-5x5.yaml',
              '--eval-task-config', c/'tasks/overfit/l1_ms-ssim.yaml', '--model-config', nvrc_model_cfg(mode, rate, wd),
              '--train-dataset-dir', DATA, '--train-dataset', SEQ, '--train-fmt', 'png', '--lamb', 1.0,
              '--start-frame', -1, *nf, '--train-video-size', -1, -1, -1, '--eval-video-size', -1, -1, -1,
              '--train-patch-size', 1, 120, 120, '--eval-patch-size', 1, -1, -1, '--grad-accum', 1, '--rate-steps', 8,
              '--train-batch-size', 144, '--eval-batch-size', 1, '--train-enable-log', 'false',
              '--eval-enable-log', 'false', '--log-epochs', -2, '--opt', 'adam', '--sched', 'cosine',
              '--auto-lr-scaling', 'true', '--max-norm', 1.0, '--workers', 4, '--prefetch-factor', 4, '--epochs', ep]
    s1 = run([*launch('nvrc', 'inductor'), *common, '--exp-config', c/'nvrc/overfit/s1-360e.yaml', '--exp-name', 's1',
              '--compress-model-config', c/'nvrc/compress_models/nvrc_s1.yaml',
              '--lr', 2e-3, '--warmup-lr', 1e-5, '--min-lr', 1e-4], NVRC, wd/'s1.log')
    s2args = ['--exp-config', c/'nvrc/overfit/s2-30e.yaml', '--compress-model-config', c/'nvrc/compress_models/nvrc_s2.yaml',
              '--lr', 1e-4, '--warmup-lr', 1e-5, '--min-lr', 1e-5]
    s2 = run([*launch('nvrc', 'inductor'), *common, *s2args, '--exp-name', 's2',
              '--resume', wd/'s1', '--resume-model-only', 'true'], NVRC, wd/'s2.log')
    dec = run([*launch('nvrc', 'no'), *common, *s2args, '--exp-name', 'dec', '--resume', wd/'s2', '--eval-only', 'true'],
              NVRC, wd/'dec.log')
    t0 = mark(dec, r'Decoding model weights from bitstream')
    t1 = mark(dec, r'Evaluation completed in')
    tdec = mark(s2, r'Decoding model weights from bitstream')
    macs_g = num(s1, r'MACs: ([\d.]+)G')
    return dict(params_M=num(s1, r'Parameters \(model\): ([\d.]+)M'),
                kmacs_px=macs_g * 1e6 / (W * H) if macs_g else None,
                epoch_s={'s1': epoch_times(s1, {None: r'Start training for'})[None],
                         's2': epoch_times(s2, {None: r'Start training for'})[None]},
                dec_s=t1 - t0, host_enc=max(s1['host_mib'], peakmon.peaks(s2['timeline'], 0, tdec)[0]),
                gpu_enc=max(s1['gpu_mib'], peakmon.peaks(s2['timeline'], 0, tdec)[1]),
                host_dec=peakmon.peaks(dec['timeline'], t0)[0], gpu_dec=peakmon.peaks(dec['timeline'], t0)[1])


# ---------------------------------------------------------------- driver
def bench(codec, mode, rate, ep):
    wd = RUNS/f'{codec.lower()}-{mode}'/rate/f'ep{ep}'
    if wd.exists():
        import shutil; shutil.rmtree(wd)
    wd.mkdir(parents=True)
    r = (hinerv if codec == 'HiNeRV' else nvrc)(mode, rate, ep, wd)
    frames, segs = (FRAMES, 1) if mode == 'full' else (SEG_FRAMES, FRAMES // SEG_FRAMES)
    enc_s = sum(SCHEDULE[codec][s] * r['epoch_s'][s][0] for s in SCHEDULE[codec])
    gpu = peakmon.subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=name', '--format=csv,noheader'],
                                 capture_output=True, text=True).stdout.strip()
    return dict(codec=codec, mode=mode, rate=rate, status='ok', gpu=gpu, frames=frames, segments=segs,
                params_M=r['params_M'], params_seq_M=r['params_M'] * segs if r['params_M'] else None,
                kmacs_px=r['kmacs_px'], gh200_kmacs_px=target_kmacs(codec, mode, rate) or '', measured_epochs=';'.join(f"{s}={r['epoch_s'][s][1]}" for s in r['epoch_s']),
                epoch_s=';'.join(f"{s}={r['epoch_s'][s][0]:.3f}" for s in r['epoch_s']),
                enc_s_segment=enc_s, enc_s_sequence=enc_s * segs, enc_fps=frames / enc_s,
                dec_s=r['dec_s'], dec_fps=frames / r['dec_s'],
                host_enc_mib=r['host_enc'], gpu_enc_mib=r['gpu_enc'], host_dec_mib=r['host_dec'], gpu_dec_mib=r['gpu_dec'],
                note=f'{ep} epochs/stage measured' + (' (1 epoch: includes compile)' if ep == 1 else ''))


def fmt(v):
    return f'{v:.4g}' if isinstance(v, float) else v


def summary(rows):
    keys = ['params_seq_M', 'kmacs_px', 'enc_fps', 'dec_fps', 'host_enc_mib', 'gpu_enc_mib', 'host_dec_mib', 'gpu_dec_mib']
    print('\nmean +- std over rate points')
    for codec in RATES:
        for mode in ['full', 'seg']:
            sel = [r for r in rows if r['codec'] == codec and r['mode'] == mode and r['status'] == 'ok']
            if not sel:
                continue
            parts = []
            for k in keys:
                v = [float(r[k]) for r in sel if r[k] not in ('', None)]
                if v:
                    parts.append(f'{k} {statistics.mean(v):.4g} +- {statistics.stdev(v) if len(v) > 1 else 0:.3g}')
            print(f'{codec:6s} {mode:4s} (n={len(sel)}): ' + ' | '.join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='HiNeRV seg s with 1 epoch per stage; prints, does not save')
    ap.add_argument('--epochs', type=int, default=3, help='epochs run per stage (first one dropped from the mean)')
    ap.add_argument('--codecs', nargs='+', default=list(RATES))
    ap.add_argument('--modes', nargs='+', default=['full', 'seg'])
    ap.add_argument('--rates', nargs='+', help='subset of rate names')
    ap.add_argument('--csv', type=Path, default=CSV)
    a = ap.parse_args()
    if a.quick:
        r = bench('HiNeRV', 'seg', 's', 1)
        print('\n'.join(f'  {k}: {fmt(v)}' for k, v in r.items()))
        summary([r])
        return
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv))) if a.csv.exists() else []
    for codec in a.codecs:
        for mode in a.modes:
            for rate in RATES[codec]:
                if a.rates and rate not in a.rates:
                    continue
                if any(r['codec'] == codec and r['mode'] == mode and r['rate'] == rate and r['status'] == 'ok' for r in rows):
                    print(f'skip {codec} {mode} {rate} (done)')
                    continue
                print(f'== {codec} {mode} {rate}', flush=True)
                t = time.time()
                try:
                    row = bench(codec, mode, rate, a.epochs)
                except Exception as e:  # keep going; the failed point is retried on the next run
                    print(f'  FAILED: {e}'.rstrip()[-2000:], flush=True)
                    row = dict(codec=codec, mode=mode, rate=rate, status='failed', note=str(e).splitlines()[0][:200])
                rows = [r for r in rows if not (r['codec'] == codec and r['mode'] == mode and r['rate'] == rate)] + [row]
                with open(a.csv, 'w', newline='') as f:
                    w = csv.DictWriter(f, FIELDS); w.writeheader()
                    w.writerows({k: fmt(r.get(k, '')) for k in FIELDS} for r in rows)
                print(f'  {row["status"]} in {time.time() - t:.0f} s: enc_fps {fmt(row.get("enc_fps", ""))} '
                      f'dec_fps {fmt(row.get("dec_fps", ""))}', flush=True)
    summary(rows)


if __name__ == '__main__':
    main()
