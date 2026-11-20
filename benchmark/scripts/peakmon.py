#!/usr/bin/env python3
"""Run a command and report wall time, peak host RAM and peak GPU memory (run alone, nothing else on the machine).

Host RAM = peak summed PSS of the process tree (psutil, sampled every --interval s); PSS splits shared pages between
processes, so forked workers (DataLoader, torch.compile) are not counted once per process as summed RSS would.
GPU memory = peak device memory.used minus the value just before the start (nvidia-smi; under WSL NVML has no
per-process numbers, so the GPU must be otherwise idle). Includes the CUDA context, unlike torch.max_memory_*.
Library use: run(cmd, marks=...) also returns the timeline, so a caller can split peaks into phases at log lines.
CLI: python scripts/peakmon.py [--gpu N] -- <command ...>
"""
import argparse, subprocess, threading, time
import psutil


def gpu_used(gpu):
    out = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def tree_rss(proc):
    try:
        return sum(p.memory_full_info().pss for p in [proc, *proc.children(recursive=True)]) / 2**20
    except psutil.Error:
        return 0.0


def run(cmd, gpu=None, interval=0.1, on_line=None, **popen):
    """Run cmd; returns dict(seconds, host_mib, gpu_mib, timeline=[(t, host_mib, gpu_mib)], lines=[(t, line)])."""
    base = gpu_used(gpu) if gpu is not None else 0.0
    t0 = time.time()
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **popen)
    ps, timeline, lines, done = psutil.Process(p.pid), [], [], threading.Event()

    def sample():
        while not done.is_set():
            timeline.append((time.time() - t0, tree_rss(ps), gpu_used(gpu) - base if gpu is not None else 0.0))
            done.wait(interval)
    th = threading.Thread(target=sample, daemon=True); th.start()
    for line in p.stdout:
        lines.append((time.time() - t0, line.rstrip()))
        if on_line:
            on_line(line.rstrip())
    p.wait(); done.set(); th.join()
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd, '\n'.join(l for _, l in lines[-30:]))
    return dict(seconds=time.time() - t0, host_mib=max((h for _, h, _ in timeline), default=0.0),
                gpu_mib=max((g for _, _, g in timeline), default=0.0), timeline=timeline, lines=lines)


def peaks(timeline, t_from=0.0, t_to=float('inf')):
    sel = [(h, g) for t, h, g in timeline if t_from <= t < t_to]
    return max((h for h, _ in sel), default=0.0), max((g for _, g in sel), default=0.0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpu', type=int)
    ap.add_argument('--interval', type=float, default=0.1)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    r = run(a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd, a.gpu, a.interval)
    print(f"wall {r['seconds']:.2f} s | peak host RAM {r['host_mib']:.0f} MiB | peak GPU memory {r['gpu_mib']:.0f} MiB")
