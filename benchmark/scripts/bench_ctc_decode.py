#!/usr/bin/env python3
"""Decoder throughput / peak RAM on 1080p for the exported CTC chunks (run alone, nothing else on the machine).

Per codec x preset x rate: decodes every chunk of the first --per-class full 1080p sequence(s) of each class,
one decoder process at a time (as bundled, default threading), output to a temp file that is deleted.
dec_fps = frames / total decoder wall time; peak_ram_mb = max RSS over all decoder processes (/usr/bin/time).
Writes <out>/ctc_decode_bench.csv (per point) and ctc_decode_bench_summary.csv (per codec x preset).
"""
import argparse, csv, os, subprocess, tempfile
NVC_ROOT = os.environ.get('NVC_ROOT', 'VideoCompression')  # sources, bitstreams, checkpoints
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--toolkit', type=Path, default=Path(NVC_ROOT, 'codecs/codec_results'))
    p.add_argument('--out', type=Path, default=Path('results/ctc_metrics'))
    p.add_argument('--codecs', nargs='+', default=['vtm', 'ecm', 'av1', 'avm'])
    p.add_argument('--presets', nargs='+', default=['ra', 'ld'])
    p.add_argument('--per-class', type=int, default=1, help='1080p sequences per class (UVG, MCL-JCV, HEVC_B, AOM-CTC-A2)')
    a = p.parse_args()

    full = {tuple(r[k] for k in ('codec', 'preset', 'class', 'sequence', 'rate'))
            for r in csv.DictReader((a.toolkit/'sequence_coverage.csv').open()) if r['full_sequence'] == 'True'}
    groups = defaultdict(list)
    for r in csv.DictReader((a.toolkit/'manifest.csv').open()):
        k = (r['codec'], r['preset'], r['sequence_class'], r['sequence'], r['rate'])
        if k in full and (r['width'], r['height']) == ('1920', '1080') and r['codec'] in a.codecs and r['preset'] in a.presets:
            groups[k].append(r)
    # Same sequences for every codec/preset: those that are complete for all of them at all rates.
    by_seq = defaultdict(set)
    for k in groups:
        by_seq[(k[2], k[3])].add((k[0], k[1], k[4]))
    need = max(len(v) for v in by_seq.values())
    picked = defaultdict(list)
    for (cls, seq), v in sorted(by_seq.items()):
        if len(v) == need and len(picked[cls]) < a.per_class:
            picked[cls].append(seq)
    seqs = {(c, s) for c, ss in picked.items() for s in ss}
    print('sequences:', ', '.join(f'{c}/{s}' for c, s in sorted(seqs)), flush=True)

    env = dict(os.environ, LD_LIBRARY_PATH=f"{a.toolkit/'decoders/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=a.out))
    rows = []
    for k in sorted(k for k in groups if (k[2], k[3]) in seqs):
        secs, peak, n = 0.0, 0, 0
        for r in sorted(groups[k], key=lambda r: int(r['first_frame'])):
            depth, part = int(r['bitdepth']), tmp/'chunk.yuv'
            dec, bs = a.toolkit/'decoders'/r['codec']/'decoder', a.toolkit/r['bitstream']
            cmd = ([dec, '--rawvideo', '--i420', f'--output-bit-depth={depth}', '-o', part, bs] if r['codec'] in ('av1', 'avm')
                   else [dec, '-b', bs, '-o', part, f'--OutputBitDepth={depth}', f'--OutputBitDepthC={depth}'])
            subprocess.run(['/usr/bin/time', '-f', '%e %M', '-o', tmp/'time', *map(str, cmd)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            e, m = (tmp/'time').read_text().split()[-2:]
            secs += float(e); peak = max(peak, int(m)); n += int(r['frames'])
            part.unlink()
        rows.append(dict(codec=k[0], preset=k[1], cls=k[2], sequence=k[3], rate=k[4], frames=n,
                         dec_seconds=secs, dec_fps=n/secs, peak_ram_mb=peak/1024))
        print(f"{' '.join(k)}: {n} frames, {n/secs:.2f} fps, {peak/1024:.0f} MB", flush=True)
    (tmp/'time').unlink(missing_ok=True); tmp.rmdir()

    with (a.out/'ctc_decode_bench.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[(r['codec'], r['preset'])].append(r)
    with (a.out/'ctc_decode_bench_summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['codec', 'preset', 'points', 'frames', 'dec_fps_1080p', 'peak_ram_mb_1080p'])
        for (c, pr), v in sorted(agg.items()):
            fr = sum(r['frames'] for r in v)
            w.writerow([c, pr, len(v), fr, fr/sum(r['dec_seconds'] for r in v), max(r['peak_ram_mb'] for r in v)])
    print(f"wrote {a.out/'ctc_decode_bench.csv'} and ctc_decode_bench_summary.csv", flush=True)


if __name__ == '__main__':
    main()
