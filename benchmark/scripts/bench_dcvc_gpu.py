#!/usr/bin/env python3
"""Throughput, peak memory and kMACs/px of DCVC-DC / -FM / -RT / -UF (LD, HT-S) and BRHVC on 1080p (run alone).

Same sequences and window as the GH200 profiling of the other learned codecs (data/profiling/learned_runs.csv):
UVG Beauty, ReadySetGo and MCL-JCV videoSRC01, videoSRC05; each model's 4 default rate points; 129 frames
(33 warm-up + 96), one I frame. Each point is the model's own test_video.py with a real bitstream (write_stream=1;
DCVC-UF always writes one). BRHVC only supports its hierarchical-B GOP 32 (I frame every 32 frames, 5 B layers),
so it codes 4 GOPs of 33 frames (frames 32, 64, 96 are coded as I twice, as test_video.py does).
  FPS     = 1000 * fpu / the "average encoding|decoding time" (ms per unit) that test_video.py prints, after the
            codec's own warm-up skip (RT 10 frames, UF 4 units); unit = P frame, B frame (BRHVC, I frames excluded)
            or 8-frame chunk (DCVC-UF HT-S, fpu=8).
  memory  = scripts/peakmon.py on the test_video.py process: host RSS of the tree, GPU memory.used above idle.
            DCVC-FM/RT/UF encode all frames and then decode them, so enc/dec are split at the first decoded frame;
            DCVC-DC and BRHVC encode and decode frame by frame, so their enc and dec peaks are the same number.
  --macs  = conv/linear/matmul MACs from torch FlopCounterMode, run in this process (use the dcvc python):
            1-frame and 2-frame runs give the I and P frame cost; kMACs/px = (I + 95 P) / 96 per frame area.
            DCVC-RT runs with its fused CUDA kernels disabled (PyTorch fallback) so the MACs are visible, decode
            subtracted (it always writes a stream), and P = mean of 32 P frames (its P cost cycles over 8); DCVC-UF
            has no PyTorch inference path, so I and P (per frame of a 2nd P unit) are counted on its training graph
            (forward_one_frame); BRHVC's 2-frame run codes nothing (GOP 32), so it uses 33 frames (2 I + 31 B,
            one GOP) and P = mean B (its own sequence average is (2 I + 31 B) / 33, not (I + 95 P) / 96).
            One model per --macs call (each repo has its own `src` package).
Writes <out>/dcvc_gpu_bench.csv (timing/memory) or <out>/dcvc_macs.json (--macs).
"""
import argparse, concurrent.futures, csv, json, multiprocessing, os, re, statistics, subprocess, sys, tempfile
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import peakmon  # noqa: E402

DCVC = Path(os.environ.get('DCVC_ROOT', 'DCVC'))
SRC = Path(NVC_ROOT, 'codecs/source_yuv')
PNG = Path(NVC_ROOT, 'BRHVC/png')  # BRHVC reads PNG only (random access); made from SRC by its own reader
SEQS = [('UVG', 'UVG/1920x1080', 'Beauty_1920x1080_120fps_420_8bit_YUV.yuv'),
        ('UVG', 'UVG/1920x1080', 'ReadySetGo_1920x1080_120fps_420_8bit_YUV.yuv'),
        ('MCL-JCV', 'MCL-JCV/1920x1080', 'videoSRC01_1920x1080_yuv420p.yuv'),
        ('MCL-JCV', 'MCL-JCV/1920x1080', 'videoSRC05_1920x1080_yuv420p.yuv')]
MODELS = {
    'dcvc-dc': dict(dir=DCVC/'DCVC-family/DCVC-DC', q=[0, 21, 42, 63], qi='--i_frame_q_indexes', qp='--p_frame_q_indexes',
                    ckpt=[f'{NVC_ROOT}/DCVC/checkpoints/DCVC-DC/cvpr2023_image_yuv420_psnr.pth.tar',
                          f'{NVC_ROOT}/DCVC/checkpoints/DCVC-DC/cvpr2023_video_yuv420_psnr.pth.tar'],
                    args=lambda c: ['--i_frame_model_path', c[0], '--p_frame_model_path', c[1], '--yuv420', '1', '--cuda', '1'],
                    seq=lambda n: dict(gop=9999)),
    # DCVC-FM-eval: the copy with the enc/dec PSNR assert relaxed (GPU kernels are not bit-exact between the passes)
    'dcvc-fm': dict(dir=Path(NVC_ROOT, 'DCVC/DCVC-FM-eval'), q=[0, 21, 42, 63], qi='--q_indexes_i', qp='--q_indexes_p',
                    ckpt=[DCVC/'checkpoints/cvpr2024_image.pth.tar', DCVC/'checkpoints/cvpr2024_video.pth.tar'],
                    args=lambda c: ['--model_path_i', c[0], '--model_path_p', c[1], '--cuda', '1'],
                    seq=lambda n: dict(intra_period=9999), split=True),  # not -1: FM tests frame_idx % intra_period == 0
    # RT's own extensions (built in-repo, not installed: the env's site-packages has DCVC-UF's of the same name)
    'dcvc-rt': dict(dir=DCVC/'DCVC-family/DCVC-RT', q=[0, 21, 42, 63], qi='--qp_i', qp='--qp_p', split=True,
                    macs_stream=1, macs_apart=[('DMCI', 'decompress'), ('DMC', 'decompress')], macs_n=33,  # P cost cycles over 8
                    ckpt=[DCVC/'checkpoints/cvpr2025_image.pth.tar', DCVC/'checkpoints/cvpr2025_video.pth.tar'],
                    path=[DCVC/'DCVC-family/DCVC-RT/src/cpp/build/lib.linux-x86_64-cpython-312',
                          DCVC/'DCVC-family/DCVC-RT/src/layers/extensions/inference/build/lib.linux-x86_64-cpython-312'],
                    args=lambda c: ['--model_path_i', c[0], '--model_path_p', c[1], '--cuda', '1',
                                    '--force_zero_thres', '0.12', '--reset_interval', '64'],  # README test command
                    seq=lambda n: dict(intra_period=-1)),
    # DCVC-UF: repo root, extensions installed in the dcvc env; README flags; intra_period -1 (9999 is not a multiple
    # of the 8-frame HT chunk and trips its assert); no --write_stream flag, it always writes the bitstream
    **{f'dcvc-uf-{s}': dict(dir=DCVC, q=[0, 21, 42, 63], qi='--qp_i', qp='--qp_p', split=True, stream=False,
                            fpu=8 if s == 'hts' else 1, quick=41 if s == 'hts' else 17,
                            ckpt=[DCVC/'checkpoints/cvpr2026_image.pth.tar', DCVC/f'checkpoints/cvpr2026_video_{s}.pth.tar'],
                            args=lambda c, s=s: ['--model_path_i', c[0], '--model_path_p', c[1], '--model_structure', s,
                                                 '--cuda_idx', '0', '--skip_thres', '0.15', '--reset_interval', '128'],
                            seq=lambda n: dict(intra_period=-1), macs=lambda d, s=s: _uf_macs(d, s))
       for s in ('ld', 'hts')},
    # BRHVC: intra model = DCVC-DC's RGB cvpr2023_image_psnr (OneDrive only); if absent, DCVC-DC's YUV420 intra
    # checkpoint (same architecture and parameter count) stands in. Explicit q 0/21/42/63 on the 64-level grid equal
    # the 4 checkpoint rate points (the grid is log-linear between them); --rate_num 1 cannot use the ckpt indexes.
    'brhvc': dict(dir=Path(__file__).resolve().parents[1]/'DVC-Review-PIEEE/submodules/BRHVC/BRHVC', q=[0, 21, 42, 63],
                  qi='--i_frame_q_indexes', qp='--b_frame_q_indexes', quick=33, macs_frames=33, macs_apart=[('IntraNoAR', 'encode_decode')],
                  ckpt=[next(p for p in [Path(NVC_ROOT, 'BRHVC/checkpoints/cvpr2023_image_psnr.pth.tar'),
                                         Path(NVC_ROOT, 'DCVC/checkpoints/DCVC-DC/cvpr2023_image_yuv420_psnr.pth.tar')]
                             if p.exists()),
                        f'{NVC_ROOT}/BRHVC/checkpoints/brhvc_model_weights.pth'],  # git-lfs object
                  args=lambda c: ['--i_frame_model_path', c[0], '--b_frame_model_path', c[1], '--cuda', '1',
                                  '--force_intra_period', '32'],
                  seq=lambda n: dict(gop=32), png=True, verbose=1),  # its verbose 2 crashes (bits[-1] is None)
}
W, H = 1920, 1080
AVG = re.compile(r'average encoding time ([\d.]+) ms, average decoding time ([\d.]+) ms')


def config(m, cls, base, seq, frames, tmp):
    png = MODELS[m].get('png')
    if png and not (PNG/base/seq[:-4]/f'im{frames:05d}.png').exists():  # test_data_to_png.py (BT.709 RGB), first frames
        subprocess.run([Path.home()/'miniforge3/envs/dcvc/bin/python', '-c', 'import sys; from src.utils.video_reader import YUVReader; '
                        'from src.utils.video_writer import PNGWriter; r = YUVReader(sys.argv[1], 1920, 1080, src_format="420"); '
                        'w = PNGWriter(sys.argv[2], 1920, 1080); [w.write_one_frame(rgb=r.read_one_frame(dst_format="rgb"), '
                        'src_format="rgb") for _ in range(int(sys.argv[3]))]', SRC/base/seq, PNG/base/seq[:-4], str(frames)],
                       cwd=MODELS[m]['dir'], check=True)
    cfg = dict(root_path=str(PNG if png else SRC), test_classes={cls: dict(
        test=1, base_path=base, src_type='png' if png else 'yuv420', sequences={
            seq[:-4] if png else seq: dict(width=W, height=H, frames=frames, **MODELS[m]['seq'](frames))})})
    (tmp/'cfg.json').write_text(json.dumps(cfg))
    return tmp/'cfg.json'


def argv(m, cfg, q, frames, stream, tmp):
    d = MODELS[m]
    return [*d['args'](d['ckpt']), '--rate_num', '1', d['qi'], q, d['qp'], q, '--force_frame_num', frames,
            *(['--write_stream', stream] if d.get('stream', True) else []), '--stream_path', tmp/'bin', '--worker', '1', '--verbose', d.get('verbose', 2),
            '--test_config', cfg, '--output_path', tmp/'out.json']


class Inline:
    """ProcessPoolExecutor stand-in that runs the work in this process (so FlopCounterMode sees it)."""
    def __init__(self, max_workers=1, initializer=None, initargs=()):
        multiprocessing.current_process().name = 'Inline-1'  # test_video.py's init_func parses the worker index from it
        if initializer:
            initializer(*initargs)

    def submit(self, fn, *a, **k):
        f = concurrent.futures.Future(); f.set_result(fn(*a, **k)); return f

    def shutdown(self, *a, **k):
        pass


def macs(a, m):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    d = MODELS[m]
    os.chdir(d['dir']); sys.path[:0] = [str(d['dir']), *map(str, d.get('path', [])[:1])]
    sys.modules['inference_extensions_cuda'] = None  # RT: no fused kernels, PyTorch fallback (UF counts its own way)
    params = sum(t.numel() for c in d['ckpt'] for t in _tensors(torch.load(c, map_location='cpu', weights_only=False)))
    cls, base, seq = SEQS[0]
    torch.distributions.Distribution.set_default_validate_args(False)  # BRHVC's entropy-estimate path trips it
    import torch.utils.module_tracker as mt  # backward-only bookkeeping; trips on RT's grad-less parameter slices
    mt.register_multi_grad_hook = lambda *x, **k: type('H', (), {'remove': lambda self: None})()
    if 'macs' in d:
        i, p = d['macs'](d)
    else:
        concurrent.futures.ProcessPoolExecutor = Inline
        multiprocessing.set_start_method = lambda *x, **k: None
        import test_video
        tmp = Path(tempfile.mkdtemp(dir=a.tmp))
        per, n_b, apart = {}, d.get('macs_frames'), []
        # methods counted apart by a nested counter: BRHVC's intra model (the I frames of its one-GOP run),
        # RT's decompress (it has no write_stream=0 path, so the decode pass is subtracted)
        for c, meth in d.get('macs_apart', []):
            def counted(self, *x, f=getattr(getattr(test_video, c), meth), **k):
                with FlopCounterMode(display=False) as fc_x:
                    r = f(self, *x, **k)
                apart.append(fc_x.get_total_flops()); return r
            setattr(getattr(test_video, c), meth, counted)
        for n in ((n_b,) if n_b else (1, d.get('macs_n', 2))):
            cfg = config(m, cls, base, seq, n, tmp)
            sys.argv = ['test_video.py', *map(str, argv(m, cfg, d['q'][1], n, d.get('macs_stream', 0), tmp))]
            with FlopCounterMode(display=False) as fc:
                test_video.main()
            per[n] = fc.get_total_flops() / 2 / (W*H) / 1e3  # kMACs per pixel, summed over the n frames
            if not n_b:
                per[n] -= sum(apart) / 2 / (W*H) / 1e3; apart.clear()
        if n_b:
            per_i = sum(apart) / 2 / (W*H) / 1e3
            i, p = per_i / len(apart), (per[n_b] - per_i) / (n_b - len(apart))
        else:
            i, p = per[1], (per[n] - per[1]) / (n - 1)
    res = dict(codec=m, sequence=seq, params_M=params/1e6, kmacs_px_I=i, kmacs_px_P=p, kmacs_px=(i + 95*p)/96,
               note='conv/linear/matmul MACs (FlopCounterMode FLOPs/2) of one test_video.py pass (encode incl. '
                    'reconstruction); entropy coding and element-wise ops not counted; params = all tensors in the checkpoints')
    out = a.out/f'dcvc_macs_{m}.json'
    out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def _tensors(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)


def _uf_macs(d, s):
    """DCVC-UF I and P kMACs/px per frame on the PyTorch training graph (its inference is a fused CUDA proxy)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from src.models.image_model import DMCI
    from src.utils.common import get_state_dict, ModelStructure
    if s == 'ld':
        from src.models.video_model_ld import DMC, g_frame_delay as fd
        p_net = DMC()
    else:
        from src.models.video_model_ht import DMC, g_frame_delay as fd
        p_net = DMC(model_structure=ModelStructure(s))
    i_net = DMCI()
    i_net.load_state_dict(get_state_dict(str(d['ckpt'][0]))); p_net.load_state_dict(get_state_dict(str(d['ckpt'][1])))
    i_net, p_net = i_net.cuda().train(), p_net.cuda().train()  # train(): the reference-feature path without the proxy
    q = torch.tensor([d['q'][1]], device='cuda')
    x = torch.rand(1, 3, (H + 15) // 16 * 16, W, device='cuda')  # padded to 16 like test_video.py
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            x_hat = i_net.forward_one_frame(x, q, recon_only=True)
        i = fc.get_total_flops() / 2 / (W*H) / 1e3
        p_net.clear_dpb(); p_net.add_ref_feature_from_frame(x_hat)
        for _ in range(2):  # the 2nd unit has the steady-state memory update (feature_adaptor_m)
            with FlopCounterMode(display=False) as fc:
                p_net.forward_one_frame(torch.cat([x] * fd, 1), q)
    return i, fc.get_total_flops() / 2 / (W*H) / 1e3 / fd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    p.add_argument('--frames', type=int, default=129)
    p.add_argument('--nseq', type=int, default=4, help='first N of SEQS (speed barely depends on content)')
    p.add_argument('--quick', action='store_true', help='one sequence, one rate, 17 frames: check that it runs')
    p.add_argument('--macs', action='store_true', help='count MACs instead (run with the dcvc python)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--dcvc-python', default=str(Path.home()/'miniforge3/envs/dcvc/bin/python'))
    p.add_argument('--out', type=Path, default=Path('results/complexity'))
    p.add_argument('--tmp', type=Path, default=Path(NVC_ROOT, 'DCVC/tmp_bench'))
    a = p.parse_args()
    a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True); a.tmp.mkdir(parents=True, exist_ok=True)
    if a.macs and os.environ.get('BENCH_MACS_CHILD'):
        return [macs(a, m) for m in a.models]

    def all_macs():  # one process per model: each repo has its own `src` package
        for m in a.models:
            if a.macs or not (a.out/f'dcvc_macs_{m}.json').exists():
                subprocess.run([a.dcvc_python, __file__, '--macs', '--models', m, '--out', a.out, '--tmp', a.tmp], env=dict(os.environ, BENCH_MACS_CHILD='1'))
    if a.macs:
        return all_macs()

    out = a.out/'dcvc_gpu_bench.csv'
    rows = list(csv.DictReader(out.open())) if out.exists() and not a.quick else []
    done = {(r['codec'], r['sequence'], r['q']) for r in rows}
    for m in a.models:
        d = MODELS[m]
        seqs, frames = (SEQS[:1], d.get('quick', 17)) if a.quick else (SEQS[:a.nseq], a.frames)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, d['path']))) if 'path' in d else None
        for cls, base, seq in seqs:
            for q in (MODELS[m]['q'][1:2] if a.quick else MODELS[m]['q']):
                if (m, seq[:-4], str(q)) in done:
                    continue
                tmp = Path(tempfile.mkdtemp(dir=a.tmp))
                cfg = config(m, cls, base, seq, frames, tmp)
                r = peakmon.run([a.dcvc_python, 'test_video.py', *argv(m, cfg, q, frames, 1, tmp)], gpu=a.gpu, cwd=d['dir'], env=env)
                avg = next((AVG.search(l) for _, l in r['lines'] if AVG.search(l)), None)
                if not avg:
                    print(f'FAIL {m} {seq} q{q}: no timing line\n' + '\n'.join(l for _, l in r['lines'][-20:]), flush=True)
                    continue
                t_dec = next((t for t, l in r['lines'] if 'decoded' in l), None) if d.get('split') else None
                he, ge = peakmon.peaks(r['timeline'], 0, t_dec or float('inf'))
                hd, gd = peakmon.peaks(r['timeline'], t_dec, float('inf')) if t_dec else (he, ge)
                row = dict(codec=m, cls=cls, sequence=seq[:-4], q=q, frames=frames, enc_fps=1000*d.get('fpu', 1)/float(avg[1]),
                           dec_fps=1000*d.get('fpu', 1)/float(avg[2]), host_enc_mib=he, host_dec_mib=hd, gpu_enc_mib=ge, gpu_dec_mib=gd,
                           wall_s=r['seconds'])
                print(' '.join(f'{k}={v:.1f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
                for f in sorted(tmp.rglob('*'), reverse=True):
                    f.unlink() if f.is_file() else f.rmdir()
                tmp.rmdir()
                if a.quick:
                    continue
                rows.append(row)
                with out.open('w', newline='') as f:
                    w = csv.DictWriter(f, list(row)); w.writeheader(); w.writerows(rows)
    for m in a.models:
        v = [r for r in rows if r['codec'] == m]
        if v:
            print(f'{m} (median of {len(v)}): ' + ', '.join(
                f"{k} {statistics.median(float(r[k]) for r in v):.1f}"
                for k in ['enc_fps', 'dec_fps', 'host_enc_mib', 'host_dec_mib', 'gpu_enc_mib', 'gpu_dec_mib']))

    if not a.quick:
        all_macs()

if __name__ == '__main__':
    main()
