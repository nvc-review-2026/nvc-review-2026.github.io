#!/usr/bin/env python3
"""Throughput, peak memory and kMACs/px of MaskCRT / B-CANF (NYCU-MAPL) on 1080p (run alone, nothing else on the GPU or CPU).

Same sequences and window as bench_dcvc_gpu.py and the GH200 profiling (data/profiling/learned_runs.csv):
UVG Beauty, ReadySetGo and MCL-JCV videoSRC01, videoSRC05; each model's 4 default rate points; 129 frames
(33 warm-up + 96 timed, counted in coding order).
  MaskCRT: one I frame, then 128 P frames (no fixed GOP needed). Released test path has no entropy coder:
           enc = the full test-mode forward (likelihoods, no bitstream); dec_fps empty. Frames are padded
           (replicate) to a multiple of 128 (1920x1152; Swin window 8 at 1/16 scale) and the model runs padded.
  B-CANF:  real bitstream (torchac, CPU arithmetic coder, in-memory) and a real decoder (test.py compress/decompress
           path). Hierarchical B with gop 16 and intra_period = frames-1 = 128, the longest the codec allows: the
           structure forces an I frame at both ends, so 129 frames hold 2 I (0, 128) + 8 B* + 119 B.
  FPS     = "core" fps: per coded frame, model input tensor (RGB on the GPU) -> bitstream/likelihoods (enc) and
            bitstream -> reconstruction (dec), torch.cuda.synchronize() around each frame; model loading, YUV read
            and YUV->RGB conversion are excluded. fps = 96 / sum(time of the 96 timed frames).
  memory  = scripts/peakmon.py on the worker process: host RSS of the tree (includes the 129 preloaded YUV frames,
            ~400 MiB), GPU memory.used above idle (incl. CUDA context); enc/dec split at the worker's "PHASE dec" line.
  --macs  = conv/linear/matmul MACs from torch FlopCounterMode (FLOPs/2), run in this process with the codec python;
            I = first frame, P = mean of the steady-state inter frames; kMACs/px = (I + 95 P)/96 over 1920x1080.
Workers run in the codec envs (--maskcrt-python, --bcanf-python); the driver only needs psutil.
Writes <out>/<codec>_gpu_bench.csv (timing/memory) or <out>/<codec>_macs.json (--macs).
"""
import argparse, csv, io, json, os, statistics, subprocess, sys, time
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUB = HERE.parent/'DVC-Review-PIEEE/submodules'
SRC = Path(NVC_ROOT, 'codecs/source_yuv')
CKPT = Path(NVC_ROOT, 'learned_ckpt')
ENVS = Path.home()/'miniforge3/envs'
SEQS = [('UVG', 'UVG/1920x1080', 'Beauty_1920x1080_120fps_420_8bit_YUV.yuv'),
        ('UVG', 'UVG/1920x1080', 'ReadySetGo_1920x1080_120fps_420_8bit_YUV.yuv'),
        ('MCL-JCV', 'MCL-JCV/1920x1080', 'videoSRC01_1920x1080_yuv420p.yuv'),
        ('MCL-JCV', 'MCL-JCV/1920x1080', 'videoSRC05_1920x1080_yuv420p.yuv')]
CODECS = {'maskcrt': dict(dir=SUB/'MaskCRT', q=[2, 3, 4, 5], ckpt=lambda q: CKPT/f'maskcrt/PSNR_Q{q}.pth.tar'),
          'bcanf': dict(dir=SUB/'B-CANF', q=[0, 1, 2, 3], ckpt=lambda q: CKPT/f'bcanf/psnr_{q}.pth')}
W, H, WARM = 1920, 1080, 33


# ---------------------------------------------------------------- worker side (codec env)
def load_frames(path, n):
    import numpy as np
    fs = W*H*3//2
    raw = np.fromfile(path, np.uint8, count=fs*n)
    return [raw[i*fs:(i+1)*fs] for i in range(n)]


def to_rgb(yuv):
    """8-bit YUV420 (BT.709 limited range) -> 1x3xHxW float RGB in [0,1] on the GPU."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(yuv).cuda().float()
    y = t[:W*H].view(1, 1, H, W)
    uv = F.interpolate(t[W*H:].view(1, 2, H//2, W//2), scale_factor=2, mode='nearest')
    y, u, v = (y - 16)/219, (uv[:, :1] - 128)/224, (uv[:, 1:] - 128)/224
    rgb = torch.cat([y + 1.5748*v, y - 0.1873*u - 0.4681*v, y + 1.8556*u], 1)
    return rgb.clamp(0, 1).contiguous()


def timed(fn, *a):
    import torch
    torch.cuda.synchronize(); t = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def maskcrt_model(q):
    import torch, yaml
    import torch.nn.functional as F
    os.chdir(CODECS['maskcrt']['dir']); sys.path.insert(0, str(CODECS['maskcrt']['dir']))
    sys.argv = sys.argv[:1]
    import CondMo_CondResTransformerCodec as M
    import trainer  # noqa: F401  (sets the cudnn flags the released test path uses: benchmark=True, tf32)
    args = M.parse_args(['--MENet', 'SPy', '--if_coder', 'DCVC-DC_Intra', '--motion_coder_conf', './config/DVC_motion.yml',
                         '--cond_motion_coder_conf', './config/CondMo_predprior.yml',
                         '--residual_coder_conf', './config/CondInter_with_CTM_Checkerboard_Context.yml',
                         '--quality_level', str(q), '--test'])
    coder = lambda f: (lambda c: M.__CODER_TYPES__[c['model_architecture']](**c['model_params']))(yaml.safe_load(open(f)))
    model = M.Pframe(args, coder(args.motion_coder_conf), coder(args.cond_motion_coder_conf),
                     coder(args.residual_coder_conf), None).cuda().eval()
    sd = torch.load(CODECS['maskcrt']['ckpt'](q), map_location='cuda')['state_dict']
    ck = {}
    for k, v in sd.items():  # same key surgery as the released main()
        k = k.split('.'); k.pop(1)
        if k[0] != 'criterion':
            ck['.'.join(k)] = v
    model.load_state_dict(ck, strict=False)
    del sd, ck

    def pad(x):
        return F.pad(x, (0, -W % 128, 0, -H % 128), mode='replicate')

    def code(idx, x):  # test_step without metrics / rate estimation
        if idx == 0:
            model.frame_buffer.clear(); model.flow_buffer.clear()
            rec = model.if_model(x)[0].clamp(0, 1)
        else:
            rec = model(model.ref, x, idx != 1)[0].clamp(0, 1)
        model.frame_buffer.append(rec)
        if len(model.frame_buffer) == 4:
            model.frame_buffer.pop(0)
        model.ref = rec
    return model, pad, code


def bcanf_model(q):
    import torch
    os.chdir(CODECS['bcanf']['dir']); sys.path.insert(0, str(CODECS['bcanf']['dir']))
    from src.models.model import BCANF
    from src.util.tools import get_setting
    torch.backends.cudnn.deterministic = True  # as test.py
    # torch>=1.5 renamed Conv2d.conv2d_forward (used by MaskedConv2d, released for torch 1.4) to _conv_forward
    torch.nn.Conv2d.conv2d_forward = lambda self, x, w: self._conv_forward(x, w, self.bias)
    model = BCANF(get_setting('./cfgs/bcanf.yaml')).cuda().eval()
    model.load_state_dict(torch.load(CODECS['bcanf']['ckpt'](q), map_location='cuda'), strict=True)
    return model


def bcanf_plan(frames):
    """Coding order of test.py: list of (display_idx, pair, mode, frame_type) for one intra period."""
    from src.datasets.dataset import get_coding_pairs
    from src.util.tools import get_order
    ip = frames - 1
    assert ip % 16 == 0 and ip <= 255, 'B-CANF needs frames-1 a multiple of 16 (gop) and <= 255'
    pairs = get_coding_pairs(ip, 16); order = get_order(pairs); plan = []
    for p in pairs:
        if len(p) == 1:
            plan.append((p[0], p, 'i-frame', -1))
        elif len(p) == 2:
            plan.append((p[1], p, 'b*-frame', 2))
        else:
            plan.append((p[1], p, 'b-frame', 0 if abs(p[-1] - p[0]) > 2 else 1))
    return plan, order


def bcanf_inputs(p, mode, buf, order, xt):
    inp = {'xt': xt}
    if mode == 'b*-frame':
        inp['x1'] = buf[p[0]]
    elif mode == 'b-frame':
        inp['x1'], inp['x2'] = buf[p[0]], buf[p[-1]]
        if order[p[0]] > order[p[-1]]:
            inp['x1'], inp['x2'] = inp['x2'], inp['x1']
    return inp


def worker(a):
    import torch
    torch.set_grad_enabled(False)
    yuv = load_frames(a.src, a.frames)
    timed_n = a.frames - WARM if a.frames > WARM else a.frames
    report = lambda tag, ts: print(f'{tag} {len(ts[-timed_n:]) / sum(ts[-timed_n:]):.6f} frames {len(ts)} '
                                   f'torch_peak_mib {torch.cuda.max_memory_allocated() / 2**20:.0f}', flush=True)
    if a.worker == 'maskcrt':
        model, pad, code = maskcrt_model(a.q)
        ts = []
        for i in range(a.frames):
            x = pad(to_rgb(yuv[i]))
            ts.append(timed(code, i, x)[1])
        report('ENC_FPS', ts)
        print('NOTE no entropy coder / decoder in the released MaskCRT test path', flush=True)
        return
    model = bcanf_model(a.q)
    from src.util.stream import Stream
    plan, order = bcanf_plan(a.frames)
    st = Stream.__new__(Stream); st.bin = io.BytesIO()
    shape = (H, W); buf = {}; ts = []
    for idx, p, mode, ft in plan:
        x = to_rgb(yuv[idx])
        def enc():
            out = model(bcanf_inputs(p, mode, buf, order, x), mode, ft, shape, 'compress')
            return out, sum(st.writeStream(s['z']) + st.writeStream(s['h']) for s in out['likelihood'])
        (out, nbytes), t = timed(enc)
        buf[idx] = out['rec_frame']; ts.append(t)
    report('ENC_FPS', ts)
    rec_enc = buf; nbits = st.bin.tell()*8
    print(f'BITS {nbits} bpp {nbits / (W*H*a.frames):.5f}', flush=True)
    print('PHASE dec', flush=True)
    st.bin.seek(0); buf = {}; ts = []
    for idx, p, mode, ft in plan:
        def dec():
            s = st.readStream()
            xt = s if mode == 'i-frame' else {'residual_stream': s, 'motion_stream': st.readStream()}
            return model(bcanf_inputs(p, mode, buf, order, xt), mode, ft, shape, 'decompress')
        out, t = timed(dec)
        buf[idx] = out['rec_frame']; ts.append(t)
    report('DEC_FPS', ts)
    mse = sum(((buf[i].clamp(0, 1) - rec_enc[i].clamp(0, 1))**2).mean().item() for i in buf) / len(buf)
    print(f'ENC_DEC_MISMATCH_MSE {mse:.3e}', flush=True)


# ---------------------------------------------------------------- MACs (codec env, in-process)
def macs(a, c):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_grad_enabled(False)
    q = CODECS[c]['q'][1]
    cls, base, seq = SEQS[0]
    yuv = load_frames(SRC/base/seq, 17)
    def count(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / 2 / (W*H) / 1e3  # kMACs per pixel of 1920x1080
    res = dict(codec=c, sequence=seq[:-4], q=q)
    if c == 'maskcrt':
        model, pad, code = maskcrt_model(q)
        per = [count(lambda: code(i, pad(to_rgb(yuv[i])))) for i in range(4)]
        i, p = per[0], statistics.mean(per[2:])
        res.update(params_M=sum(t.numel() for t in model.parameters())/1e6, kmacs_px_I=i, kmacs_px_P=p,
                   kmacs_px_P_first=per[1], kmacs_px=(i + 95*p)/96,
                   note='conv/linear/matmul MACs (FlopCounterMode FLOPs/2) of the test-mode forward on the 1920x1152 '
                        'padded frame, divided by 1920x1080; P = steady-state P (frames 2,3; frame 1 has no motion '
                        'prediction: kmacs_px_P_first); no entropy coder in the released path, no separate decoder; '
                        'element-wise ops and warping not counted; params = model.parameters()')
    else:
        model = bcanf_model(q)
        from src.util.stream import Stream
        plan, order = bcanf_plan(17)
        st = Stream.__new__(Stream); st.bin = io.BytesIO(); buf = {}; enc = {}
        for idx, p, mode, ft in plan:
            x = to_rgb(yuv[idx]); o = {}
            enc[idx] = count(lambda: o.update(model(bcanf_inputs(p, mode, buf, order, x), mode, ft, (H, W), 'compress')))
            [st.writeStream(s['z']) + st.writeStream(s['h']) for s in o['likelihood']]
            buf[idx] = o['rec_frame']
        st.bin.seek(0); buf = {}; dec = {}
        for idx, p, mode, ft in plan:
            o = {}
            s = st.readStream()
            xt = s if mode == 'i-frame' else {'residual_stream': s, 'motion_stream': st.readStream()}
            dec[idx] = count(lambda: o.update(model(bcanf_inputs(p, mode, buf, order, xt), mode, ft, (H, W), 'decompress')))
            buf[idx] = o['rec_frame']
        inter = [idx for idx, _, m, _ in plan if m != 'i-frame']
        i, p = enc[0], statistics.mean(enc[k] for k in inter)
        di, dp = dec[0], statistics.mean(dec[k] for k in inter)
        res.update(params_M=sum(t.numel() for t in model.parameters())/1e6, kmacs_px_I=i, kmacs_px_P=p, kmacs_px=(i + 95*p)/96,
                   kmacs_px_dec_I=di, kmacs_px_dec_P=dp, kmacs_px_dec=(di + 95*dp)/96,
                   note='conv/linear/matmul MACs (FlopCounterMode FLOPs/2) of compress (enc) and decompress (dec) on '
                        '1920x1080 (internal padding included); P = mean over the B*/B frames of a 17-frame '
                        'intra period (gop 16); entropy coding (torchac) and element-wise ops not counted; '
                        'params = model.parameters()')
    out = a.out/f'{c}_macs.json'
    out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--codecs', nargs='+', default=list(CODECS), choices=list(CODECS))
    p.add_argument('--frames', type=int, default=129)
    p.add_argument('--nseq', type=int, default=4, help='first N of SEQS (speed barely depends on content)')
    p.add_argument('--quick', action='store_true', help='one sequence, one rate, 17 frames: check that it runs')
    p.add_argument('--macs', action='store_true', help='count MACs instead (run with the codec python, one codec)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--maskcrt-python', default=str(ENVS/'maskcrt/bin/python'))
    p.add_argument('--bcanf-python', default=str(ENVS/'bcanf/bin/python'))
    p.add_argument('--out', type=Path, default=Path('results/complexity'))
    p.add_argument('--worker', choices=list(CODECS), help=argparse.SUPPRESS)
    p.add_argument('--src', help=argparse.SUPPRESS)
    p.add_argument('--q', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        return worker(a)
    a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    if a.macs and os.environ.get('BENCH_MACS_CHILD'):
        return [macs(a, c) for c in a.codecs]

    def all_macs():  # in each codec's own env
        for c in a.codecs:
            if a.macs or not (a.out/f'{c}_macs.json').exists():
                subprocess.run([getattr(a, f'{c}_python'), __file__, '--macs', '--codecs', c, '--out', a.out], env=dict(os.environ, BENCH_MACS_CHILD='1'))
    if a.macs:
        return all_macs()

    sys.path.insert(0, str(HERE))
    import peakmon
    seqs, frames = (SEQS[:1], 17) if a.quick else (SEQS[:a.nseq], a.frames)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_VISIBLE_DEVICES=str(a.gpu))
    for c in a.codecs:
        out = a.out/f'{c}_gpu_bench.csv'
        rows = list(csv.DictReader(out.open())) if out.exists() and not a.quick else []
        done = {(r['sequence'], r['q']) for r in rows}
        for cls, base, seq in seqs:
            for q in (CODECS[c]['q'][1:2] if a.quick else CODECS[c]['q']):
                if (seq[:-4], str(q)) in done:
                    continue
                cmd = [getattr(a, f'{c}_python'), __file__, '--worker', c, '--src', SRC/base/seq, '--q', q, '--frames', frames]
                try:
                    r = peakmon.run(cmd, gpu=a.gpu, env=env)
                except Exception as e:
                    print(f'FAIL {c} {seq} q{q}: {e}\n{getattr(e, "output", "")}', flush=True)
                    continue
                # tqdm progress bars (\r, no newline) can prefix our lines, so search inside each line
                get = lambda tag: next((l[l.index(tag):].split() for _, l in r['lines'] if tag in l), None)
                enc, dec = get('ENC_FPS'), get('DEC_FPS')
                t_dec = next((t for t, l in r['lines'] if 'PHASE dec' in l), None)
                he, ge = peakmon.peaks(r['timeline'], 0, t_dec or float('inf'))
                hd, gd = peakmon.peaks(r['timeline'], t_dec) if t_dec else ('', '')
                row = dict(codec=c, cls=cls, sequence=seq[:-4], q=q, frames=frames, enc_fps=float(enc[1]),
                           dec_fps=float(dec[1]) if dec else '', host_enc_mib=he, host_dec_mib=hd,
                           gpu_enc_mib=ge, gpu_dec_mib=gd, wall_s=r['seconds'])
                print(' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
                for tag in ('ENC_FPS', 'DEC_FPS', 'BITS', 'ENC_DEC_MISMATCH_MSE', 'NOTE'):
                    if get(tag):
                        print('  ' + ' '.join(get(tag)), flush=True)
                if a.quick:
                    continue
                rows.append(row)
                with out.open('w', newline='') as f:
                    w = csv.DictWriter(f, list(row)); w.writeheader(); w.writerows(rows)
        v = [r for r in rows if r['codec'] == c]
        if v:
            print(f'{c} (median of {len(v)}): ' + ', '.join(
                f"{k} {statistics.median(float(r[k]) for r in v):.2f}"
                for k in ['enc_fps', 'dec_fps', 'host_enc_mib', 'host_dec_mib', 'gpu_enc_mib', 'gpu_dec_mib'] if v[0][k] != ''))

    if not a.quick:
        all_macs()

if __name__ == '__main__':
    main()
