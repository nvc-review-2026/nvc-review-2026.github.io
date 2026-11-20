#!/usr/bin/env python3
"""96-frame UVG/HEVC-B PNG benchmark: CTC QPs, RGB metrics, RGB-derived VMAF."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

ROOT = Path(os.environ.get('PNG_ROOT', 'PNG/BT.601'))


def run(cmd, log, cwd=None):
    cmd = list(map(str, cmd))
    print(time.strftime('%H:%M:%S'), ' '.join(cmd), flush=True)
    with Path(log).open('w') as f:
        f.write(json.dumps(cmd) + '\n'); f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=cwd)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def natural(p):
    return [int(v) if v.isdigit() else v for v in re.split(r'(\d+)', p.name)]


def discover(a):
    from PIL import Image
    sequences = []
    groups = [('UVG', a.root / 'UVG' / a.uvg_resolution),
              ('HEVC-B', a.root / 'HEVC-CTC' / 'B')]
    for dataset, group in groups:
        if dataset not in a.datasets:
            continue
        if not group.is_dir():
            raise ValueError(f'Missing dataset directory: {group}')
        for folder in sorted(group.iterdir()):
            if not folder.is_dir() or (a.sequence and a.sequence != folder.name):
                continue
            images = sorted(folder.glob('*.png'), key=natural)[:a.frames]
            if len(images) != a.frames:
                raise ValueError(f'{folder}: requires {a.frames} PNGs, found {len(images)}')
            with Image.open(images[0]) as im:
                w, h = im.size
            if w % 2 or h % 2:
                raise ValueError('YUV420 requires even dimensions')
            # Original UVG ShakeNDry is 120 fps too; no blanket HEVC fps assumption.
            fps = 120 if dataset == 'UVG' else int(folder.name.rsplit('_', 1)[1])
            sequences.append(dict(dataset=dataset, sequence=folder.name, width=w, height=h,
                                  fps=fps, images=[str(p.resolve()) for p in images]))
    if not sequences:
        raise ValueError('No matching sequences')
    return sequences


def raw_input(path, s, fmt):
    return ['-f', 'rawvideo', '-pixel_format', fmt, '-video_size',
            f"{s['width']}x{s['height']}", '-framerate', str(s['fps']), '-i', str(path)]


def period(fps, mode, gop):
    if mode == 'AI':
        return 1
    if mode == 'LD':
        return -1
    # JVET RA GOP32 schedule; 120 fps is the UVG extension.
    ip = {24: 32, 25: 32, 30: 32, 50: 64, 60: 64, 100: 96, 120: 128}[fps]
    if gop != 32:
        raise ValueError(f'RA requires the complete GOP32 VTM config, got GOPSize={gop}. '
                         'Use the matching release config with all 32 Frame entries; '
                         'do not just override GOPSize in a GOP16 config.')
    return ip


def metrics(a, s, reference, reconstruction, directory, models):
    import numpy as np
    import torch
    lpips_model, dists_model = models
    w, h = s['width'], s['height']
    shape = (a.frames, h, w, 3)
    ref = np.memmap(reference, mode='r', dtype=np.uint8, shape=shape)
    dec = np.memmap(reconstruction, mode='r', dtype=np.uint8, shape=shape)
    rows = []
    with torch.inference_mode():
        for i in range(a.frames):
            x = torch.from_numpy(np.array(ref[i])).permute(2, 0, 1).unsqueeze(0).to(a.device).float()/255
            y = torch.from_numpy(np.array(dec[i])).permute(2, 0, 1).unsqueeze(0).to(a.device).float()/255
            mse = (x-y).square().double().mean().item()
            rows.append(dict(frame=i, mse_rgb=mse,
                             psnr_rgb=10*math.log10(1/mse) if mse else float('inf'),
                             lpips=lpips_model(x*2-1, y*2-1).item(),
                             dists=dists_model(x, y).item()))
            if i % 8 == 0:
                print(f"  RGB metrics {i+1}/{a.frames}", flush=True)
    # Standard VMAF consumes luma. Both inputs originate from the exact RGB metric pair.
    conversion = f'scale=in_range=full:out_range=limited:out_color_matrix={a.matrix}:flags=bicubic,format=yuv444p,setpts=PTS-STARTPTS'
    graph = (f'[0:v]{conversion}[d];[1:v]{conversion}[r];'
             '[d][r]libvmaf=model=version=vmaf_v0.6.1:log_fmt=json:log_path=vmaf.json'
             f':n_threads={a.threads}')
    run([a.vmaf_ffmpeg, '-nostdin', '-v', 'info', *raw_input(reconstruction, s, 'rgb24'),
         *raw_input(reference, s, 'rgb24'), '-filter_complex', graph,
         '-frames:v', a.frames, '-f', 'null', '-'], directory/'vmaf.log', cwd=directory)
    vmaf = json.loads((directory/'vmaf.json').read_text())
    if len(vmaf['frames']) != a.frames:
        raise ValueError('VMAF frame count mismatch')
    for row, v in zip(rows, vmaf['frames']):
        row['vmaf_rgb_luma'] = v['metrics']['vmaf']
    with (directory/'frames.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    mse = sum(r['mse_rgb'] for r in rows)/a.frames
    return dict(psnr_rgb=10*math.log10(1/mse) if mse else float('inf'),
                psnr_rgb_frame_mean=sum(r['psnr_rgb'] for r in rows)/a.frames,
                lpips=sum(r['lpips'] for r in rows)/a.frames,
                dists=sum(r['dists'] for r in rows)/a.frames,
                vmaf_rgb_luma=sum(r['vmaf_rgb_luma'] for r in rows)/a.frames)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, default=Path('results/x265_vtm_rgb_96'))
    p.add_argument('--datasets', nargs='+', choices=['UVG', 'HEVC-B'], default=['UVG', 'HEVC-B'])
    p.add_argument('--uvg-resolution', default='1920x1080')
    p.add_argument('--sequence')
    p.add_argument('--frames', type=int, default=96)
    p.add_argument('--qps', nargs='+', type=int, default=[22, 27, 32, 37])
    p.add_argument('--mode', choices=['RA', 'LD', 'AI'], default='RA')
    p.add_argument('--codecs', nargs='+', choices=['x265', 'vtm'], default=['x265', 'vtm'])
    p.add_argument('--ffmpeg', default='/usr/bin/ffmpeg', help='FFmpeg with libx265 encoder')
    p.add_argument('--vmaf-ffmpeg', default=os.environ.get('VMAF_FFMPEG', 'ffmpeg'))
    p.add_argument('--vtm-root', type=Path, help='Built VTM release containing cfg/ and bin/')
    p.add_argument('--vtm-encoder', type=Path)
    p.add_argument('--vtm-decoder', type=Path)
    p.add_argument('--vtm-config', type=Path, help='Matching release CTC config; RA protocol requires GOP32')
    p.add_argument('--preset', default='veryslow', help='x265 preset')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='auto', help='auto, cuda, cuda:0, or cpu')
    p.add_argument('--matrix', choices=['bt601', 'bt709'], default='bt601')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.threads < 1 or a.frames < 1 or not a.qps or any(q < 0 or q > 51 for q in a.qps):
        p.error('Invalid frames or QPs')
    a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    sequences = discover(a)
    gop = 32 if a.mode == 'RA' else 1
    config_text = ''
    if a.vtm_root:
        a.vtm_encoder = a.vtm_encoder or a.vtm_root/'bin/EncoderAppStatic'
        a.vtm_decoder = a.vtm_decoder or a.vtm_root/'bin/DecoderAppStatic'
        name = {'RA': 'randomaccess', 'LD': 'lowdelay', 'AI': 'intra'}[a.mode]
        a.vtm_config = a.vtm_config or a.vtm_root/f'cfg/encoder_{name}_vtm.cfg'
    if a.vtm_config:
        a.vtm_config = a.vtm_config.resolve()
        config_text = a.vtm_config.read_text()
        match = re.search(r'^\s*GOPSize\s*:\s*(\d+)', config_text, re.M)
        if not match:
            p.error('GOPSize missing from VTM config')
        gop = int(match[1])
    for s in sequences:
        s['intra_period'] = period(s['fps'], a.mode, gop)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                if k not in ['plan_only', 'out']}
    manifest = dict(settings=settings, sequences=sequences, vtm_config=config_text,
                    protocol='CTC-derived, first N PNG frames continuously encoded; not full official CTC',
                    rgb='original PNG RGB vs decoded RGB, full resolution, no resizing/cropping',
                    vmaf='vmaf_v0.6.1 on limited-range luma derived from the RGB pair',
                    coding='10-bit 4:2:0; VTM RA GOP32; x265 mini-GOP16 baseline with matched intra periods')
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    print(f'{len(sequences)} sequences, {len(sequences)*len(a.codecs)*len(a.qps)} jobs; {a.mode}, QPs={a.qps}', flush=True)
    if a.plan_only:
        atomic_json(a.out/'plan.json', manifest)
        return
    if 'vtm' in a.codecs:
        for item in [a.vtm_encoder, a.vtm_decoder, a.vtm_config]:
            if item is None or not item.is_file():
                p.error('VTM is required: supply --vtm-root or --vtm-encoder/--vtm-decoder/--vtm-config')
        a.vtm_encoder = a.vtm_encoder.resolve(); a.vtm_decoder = a.vtm_decoder.resolve()
    old = a.out/'manifest.json'
    if old.exists() and json.loads(old.read_text()).get('signature') != signature:
        p.error('Output directory belongs to different settings; choose a new --out')
    encoders = subprocess.check_output([a.ffmpeg, '-hide_banner', '-encoders'], stderr=subprocess.STDOUT).decode()
    if 'x265' in a.codecs and 'libx265' not in encoders:
        p.error('--ffmpeg does not provide libx265')
    filters = subprocess.check_output([a.vmaf_ffmpeg, '-hide_banner', '-filters'], stderr=subprocess.STDOUT).decode()
    if 'libvmaf' not in filters:
        p.error('--vmaf-ffmpeg does not provide libvmaf')
    import torch
    import lpips
    from DISTS_pytorch import DISTS
    torch.set_num_threads(a.threads)
    a.device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    print(f'Loading LPIPS/AlexNet and DISTS on {a.device}', flush=True)
    models = (lpips.LPIPS(net='alex').eval().to(a.device), DISTS().eval().to(a.device))
    # Fail before encoding if pretrained weights/device are unusable.
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 64, 64, device=a.device)
        for model in models:
            if not torch.isfinite(model(probe, probe)).all():
                raise ValueError('Nonfinite metric preflight')
    atomic_json(old, manifest | dict(signature=signature, actual_device=a.device))
    run([a.ffmpeg, '-version'], a.out/'ffmpeg_version.log')
    run([a.vmaf_ffmpeg, '-version'], a.out/'vmaf_ffmpeg_version.log')
    rows = []
    for s in sequences:
        with tempfile.TemporaryDirectory(prefix='rgb-bench-', dir=a.out) as tmp:
            work = Path(tmp)
            ref = work/'reference.rgb'; source = work/'source.yuv'
            pending = [(c, q, a.out/s['dataset']/s['sequence']/c/a.mode/f'qp{q}')
                       for c in a.codecs for q in a.qps]
            if any(not (d/'result.json').exists() for _, _, d in pending):
                from PIL import Image
                with ref.open('wb') as f:
                    for name in s['images']:
                        with Image.open(name) as im:
                            if im.size != (s['width'], s['height']):
                                raise ValueError(f'Inconsistent frame size: {name}')
                            f.write(im.convert('RGB').tobytes())
                run([a.ffmpeg, '-nostdin', '-y', '-v', 'warning', *raw_input(ref, s, 'rgb24'),
                     '-vf', f'scale=in_range=full:out_range=limited:out_color_matrix={a.matrix}:flags=bicubic',
                     '-frames:v', a.frames, '-pix_fmt', 'yuv420p10le', '-f', 'rawvideo', source], work/'convert.log')
                shutil.copy(work/'convert.log', a.out/f"{s['dataset']}_{s['sequence']}_convert.log")
                if source.stat().st_size != s['width']*s['height']*3*a.frames:
                    raise ValueError('Source frame count mismatch')
            for codec, qp, d in pending:
                d.mkdir(parents=True, exist_ok=True)
                result_path = d/'result.json'
                if result_path.exists():
                    row = json.loads(result_path.read_text())
                else:
                    bits = d/('stream.hevc' if codec == 'x265' else 'stream.vvc')
                    decoded = work/'decoded.yuv'; rgb = work/'decoded.rgb'
                    if codec == 'x265':
                        ip = s['intra_period'] if a.mode != 'LD' else 2147483647
                        params = (f'qp={qp}:keyint={ip}:min-keyint={ip}:scenecut=0:b-adapt=0:'
                                  f'bframes={15 if a.mode == "RA" else 0}:open-gop=1:aq-mode=0:cutree=0:'
                                  f'rc-lookahead={40 if a.mode == "RA" else 0}:b-pyramid=1:'
                                  f'pools={a.threads}:frame-threads=1:range=limited')
                        cmd = [a.ffmpeg, '-nostdin', '-y', *raw_input(source, s, 'yuv420p10le'),
                               '-frames:v', a.frames, '-c:v', 'libx265', '-preset', a.preset, '-tune', 'psnr',
                               '-pix_fmt', 'yuv420p10le', '-x265-params', params, '-f', 'hevc', bits]
                    else:
                        cmd = [a.vtm_encoder, '-c', a.vtm_config, '-i', source, '-b', bits,
                               '-o', os.devnull, '-wdt', s['width'], '-hgt', s['height'], '-fr', s['fps'],
                               '-f', a.frames, '-q', qp, '-ip', s['intra_period'], '--InputBitDepth=10',
                               '--InternalBitDepth=10', '--OutputBitDepth=10', '--InputChromaFormat=420',
                               '--FrameSkip=0', '--TemporalSubsampleRatio=1']
                    started = time.monotonic(); run(cmd, d/'encode.log'); enc_seconds = time.monotonic()-started
                    if codec == 'x265':
                        cmd = [a.ffmpeg, '-nostdin', '-y', '-i', bits, '-pix_fmt', 'yuv420p10le', '-f', 'rawvideo', decoded]
                    else:
                        cmd = [a.vtm_decoder, '-b', bits, '-o', decoded, '--OutputBitDepth=10']
                    run(cmd, d/'decode.log')
                    if decoded.stat().st_size != s['width']*s['height']*3*a.frames:
                        raise ValueError('Decoded frame count mismatch')
                    run([a.ffmpeg, '-nostdin', '-y', *raw_input(decoded, s, 'yuv420p10le'),
                         '-vf', f'scale=in_range=limited:out_range=full:in_color_matrix={a.matrix}:flags=bicubic',
                         '-frames:v', a.frames, '-pix_fmt', 'rgb24', '-f', 'rawvideo', rgb], d/'rgb.log')
                    if rgb.stat().st_size != s['width']*s['height']*3*a.frames:
                        raise ValueError('RGB frame count mismatch')
                    scores = metrics(a, s, ref, rgb, d, models)
                    size = bits.stat().st_size
                    row = {k: s[k] for k in ['dataset', 'sequence', 'width', 'height', 'fps', 'intra_period']}
                    row.update(vtm_gop_size=gop if codec == 'vtm' else None,
                               x265_bframes=(15 if a.mode == 'RA' else 0) if codec == 'x265' else None,
                               codec=codec, mode=a.mode, qp=qp, frames=a.frames, bytes=size,
                               bitrate_kbps=size*8*s['fps']/a.frames/1000,
                               bpp=size*8/(a.frames*s['width']*s['height']), encode_seconds=enc_seconds, **scores)
                    atomic_json(result_path, row)
                rows.append(row)
                tmp_csv = a.out/'results.csv.tmp'
                with tmp_csv.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                tmp_csv.replace(a.out/'results.csv')
                print(f"DONE {s['sequence']} {codec} QP {qp}: {row['bitrate_kbps']:.2f} kbps", flush=True)
    print(f'Results: {a.out / "results.csv"}', flush=True)


if __name__ == '__main__':
    main()
