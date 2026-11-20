#!/usr/bin/env python3
"""Independent chunk codec benchmark; resume completed jobs, retain commands and metrics."""
import argparse, csv, json, os, re, subprocess, time, threading, math
from pathlib import Path
from collections import defaultdict

ROOT = os.environ.get('PNG_ROOT', 'PNG/BT.601')
HOME = Path.home()
FF = str(HOME / '.local/ffmpeg-nvenc/bin/ffmpeg')
VE = str(HOME / '.local/bin/vvencapp')
VD = str(HOME / '.local/bin/vvdecapp')
ENV = dict(os.environ)
ENV['LD_LIBRARY_PATH'] = ':'.join([str(HOME / '.local/ffmpeg-nvenc/lib'), '/usr/lib/wsl/lib', ENV.get('LD_LIBRARY_PATH','')])

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

def run(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w') as f:
        f.write(json.dumps(cmd)+'\n'); f.flush()
        t = time.monotonic()
        subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
    return time.monotonic()-t

def natural(p):
    return [int(x) if x.isdigit() else x for x in re.split('(\\d+)', p.name)]

def fps_for(dataset, name):
    if dataset == 'UVG': return '120', 'dataset default'
    if dataset == 'MCL-JCV': return '30', 'assumed; PNG files contain no timing'
    if '5994' in name: return '60000/1001', 'sequence name'
    if '2997' in name: return '30000/1001', 'sequence name'
    m = re.search(r'(?:p|_)(24|25|30|50|60)(?:fps|p|_|$)', name)
    return (m.group(1), 'sequence name') if m else ('30', 'assumed; PNG files contain no timing')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=ROOT)
    p.add_argument('--out', type=Path, default=Path('results/png_bt601_gop32_96'))
    p.add_argument('--codecs', nargs='+', choices=['vvenc','nvenc'], default=['vvenc','nvenc'])
    p.add_argument('--datasets', nargs='+', default=['UVG','MCL-JCV','HEVC-CTC','AOM-CTC'])
    p.add_argument('--qps', nargs='+', type=int, default=[22,27,32,37])
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--frames', type=int, default=96)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--preset', default='medium')
    p.add_argument('--sequence', help='optional exact sequence name for smoke checks')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.frames % a.chunk: p.error('--frames must be divisible by --chunk')
    a.out = a.out.resolve(); a.out.mkdir(parents=True,exist_ok=True)
    sequences=[]; excluded=[]
    for dataset in a.datasets:
        for directory in sorted((Path(a.root)/dataset).glob('*/*')):
            if not directory.is_dir() or (a.sequence and directory.name != a.sequence): continue
            images=sorted(directory.glob('*.png'),key=natural)
            if len(images)<a.frames:
                excluded.append({'sequence':str(directory),'available_frames':len(images),'required_frames':a.frames})
                log(f'SKIP insufficient frames: {directory} ({len(images)} < {a.frames})'); continue
            fps,basis=fps_for(dataset,directory.name)
            sequences.append(dict(dataset=dataset, group=directory.parent.name, sequence=directory.name,
                                  directory=str(directory), images=[str(x) for x in images[:a.frames]], fps=fps, fps_basis=basis))
    plan={'settings':vars(a)|{'out':str(a.out)},'sequences':sequences,'excluded':excluded,
          'input_format':'8-bit yuv420p, BT.601 limited range',
          'metrics':'FFmpeg PSNR and SSIM on 8-bit YUV420 reference and decoded frames',
          'LD':'VVenC PicReordering=0 GOPSize=8; NVENC bf=0 lookahead=0 zerolatency=1',
          'RA':'VVenC hierarchical GOPSize=32 PicReordering=1; NVENC bf=3 b_ref_mode=middle (hardware RA, not VVC-equivalent hierarchy)',
          'chunks':'each chunk encoded and decoded independently, first picture IDR',
          'fps_note':'PNG input has no timestamps; fps assumptions affect rate/runtime ratios, not bpp or PSNR/SSIM'}
    (a.out/'manifest.json').write_text(json.dumps(plan,indent=2))
    log(f'PLAN {len(sequences)} sequences; {len(excluded)} excluded; {len(sequences)*2*len(a.qps)*(a.frames//a.chunk)*len(a.codecs)} chunk jobs')
    if a.plan_only: return
    blocked=[]; codecs=list(a.codecs)
    if 'nvenc' in codecs:
        try:
            run([FF,'-nostdin','-y','-f','lavfi','-i','color=size=640x360','-frames:v','2','-c:v','hevc_nvenc','-f','hevc',str(a.out/'nvenc_probe.hevc')],a.out/'nvenc_probe.log')
        except subprocess.CalledProcessError as e:
            blocked.append({'codec':'nvenc','reason':f'NVENC preflight failed: exit {e.returncode}; see nvenc_probe.log'})
            codecs.remove('nvenc'); log('BLOCKED NVENC GPU preflight; proceeding with available codecs')
    (a.out/'blocked.json').write_text(json.dumps(blocked,indent=2))
    if not codecs: raise RuntimeError('No available requested codec')
    tasks=[]
    for s in sequences:
        for codec in codecs:
            for mode in ['LD','RA']:
                for qp in a.qps:
                    for chunk in range(a.frames//a.chunk):
                        directory=a.out/s['dataset']/s['group']/s['sequence']/codec/mode/f'q{qp}'/f'chunk{chunk:02}'
                        tasks.append((s,codec,mode,qp,chunk,directory))
    def export_results():
        rows=[json.loads((d/'result.json').read_text()) for *_,d in tasks if (d/'result.json').exists()]
        if not rows: return
        def write_csv(name, items):
            temp=a.out/(name+'.tmp')
            with temp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(items[0])); writer.writeheader(); writer.writerows(items)
            temp.replace(a.out/name)
        write_csv('results_chunks.csv',rows)
        groups=defaultdict(list)
        for row in rows: groups[tuple(row[k] for k in ['dataset','group','sequence','codec','mode','qp'])].append(row)
        summary=[]
        for key, rr in groups.items():
            item=dict(zip(['dataset','group','sequence','codec','mode','qp'],key))
            item.update(chunks_completed=len(rr),chunks_expected=a.frames//a.chunk,complete=len(rr)==a.frames//a.chunk,frames=sum(r['frames'] for r in rr),bytes=sum(r['bytes'] for r in rr),bpp=sum(r['bpp'] for r in rr)/len(rr),enc_seconds=sum(r['enc_seconds'] for r in rr),dec_seconds=sum(r['dec_seconds'] for r in rr))
            for metric in ['psnr_y','psnr_u','psnr_v','psnr_avg']:
                mse=sum(10**(-r[metric]/10) for r in rr)/len(rr)
                item[metric]=-10*math.log10(mse) if mse else float('inf')
            item['ssim_all']=sum(r['ssim_all'] for r in rr)/len(rr)
            summary.append(item)
        write_csv('results_sequences.csv',summary)
    export_results()
    done=sum((d/'result.json').exists() for *_,d in tasks)
    timings=defaultdict(list); state={'done':done,'total':len(tasks),'current':'initializing'}; start=time.monotonic()
    stop=threading.Event()
    def progress():
        # Work weighted by pixels; separate timing models for each codec and mode.
        remaining=[t for t in tasks if not (t[-1]/'result.json').exists()]
        eta=0; unknown=False
        for s,c,m,q,k,d in remaining:
            model=timings[(c,m)]
            if not model:
                model=[v for (cc,mm),values in timings.items() if cc==c for v in values]
                unknown=True
            if not model: continue
            wh=re.search(r'(\d+)x(\d+)',s['group']+'_'+s['sequence'])
            pixels=int(wh[1])*int(wh[2]) if wh else 1920*1080
            eta+=sum(model)/len(model)*pixels*a.chunk
        value=('calibrating; waiting for first completed chunk' if not any(timings.values()) else (f'{eta/3600:.2f} hours' if not unknown else f'provisional {eta/3600:.2f} hours (calibrating codec/mode timings)'))
        log(f"PROGRESS {state['done']}/{state['total']} elapsed={(time.monotonic()-start)/3600:.2f}h estimated_remaining={value}; current={state['current']}")
    def heartbeat():
        while not stop.wait(30): progress()
    threading.Thread(target=heartbeat,daemon=True).start()
    failures=[]
    try:
        for s in sequences:
            sequence_tasks=[t for t in tasks if t[0] is s and not (t[-1]/'result.json').exists()]
            if not sequence_tasks: continue
            work=a.out/'work'/s['dataset']/s['group']/s['sequence']; work.mkdir(parents=True,exist_ok=True)
            # Symlinks provide an exact, naturally sorted first-N input list without concat timestamp rounding.
            for i,src in enumerate(s['images']):
                link=work/f'{i:06}.png'
                if not link.exists(): link.symlink_to(src)
            info=subprocess.check_output([str(HOME/'.local/ffmpeg-nvenc/bin/ffprobe'),'-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',s['images'][0]],env=ENV)
            stream=json.loads(info)['streams'][0]; w,h=stream['width'],stream['height']
            if w%2 or h%2: raise ValueError(f'Odd dimensions: {s["sequence"]}')
            reference=work/'reference.yuv'; state['current']=f'convert {s["dataset"]}/{s["sequence"]}'
            run([FF,'-nostdin','-y','-v','warning','-framerate',s['fps'],'-i',str(work/'%06d.png'),'-frames:v',str(a.frames),'-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt601:flags=bicubic,format=yuv420p','-pix_fmt','yuv420p','-f','rawvideo',str(reference)],work/'convert.log')
            framebytes=w*h*3//2
            if reference.stat().st_size != framebytes*a.frames: raise ValueError('Reference frame count mismatch')
            with reference.open('rb') as f:
                for k in range(a.frames//a.chunk): (work/f'chunk{k:02}.yuv').write_bytes(f.read(framebytes*a.chunk))
            for _,codec,mode,qp,k,d in sequence_tasks:
                d.mkdir(parents=True,exist_ok=True); jobstart=time.monotonic()
                state['current']=f'{s["dataset"]}/{s["sequence"]} {codec} {mode} QP={qp} chunk={k}'
                raw=work/f'chunk{k:02}.yuv'; bits=d/('stream.266' if codec=='vvenc' else 'stream.hevc'); dec=d/'decoded.yuv'
                try:
                    if codec=='vvenc':
                        extra=f'GOPSize={8 if mode=="LD" else 32}:PicReordering={0 if mode=="LD" else 1}:POC0IDR=1:InternalBitDepth=10'
                        enc=[VE,'-i',str(raw),'-s',f'{w}x{h}','--fps',s['fps'],'--format','yuv420','--frames',str(a.chunk),'--preset',a.preset,'--threads',str(a.threads),'--qpa','0','--qp',str(qp),'--intraperiod',str(a.chunk),'--refreshtype','none' if mode=='LD' else 'idr','--additional',extra,'-o',str(bits)]
                        es=run(enc,d/'encode.log')
                        intermediate=d/'decoded.y4m'
                        ds=run([VD,'-b',str(bits),'-o',str(intermediate),'-t',str(a.threads)],d/'decode.log')
                        run([FF,'-nostdin','-y','-v','warning','-i',str(intermediate),'-pix_fmt','yuv420p','-f','rawvideo',str(dec)],d/'normalize.log')
                        intermediate.unlink()
                    else:
                        enc=[FF,'-nostdin','-y','-v','info','-f','rawvideo','-pix_fmt','yuv420p','-s',f'{w}x{h}','-r',s['fps'],'-i',str(raw),'-frames:v',str(a.chunk),'-c:v','hevc_nvenc','-preset','p5','-rc','constqp','-qp',str(qp),'-g',str(a.chunk),'-bf','0' if mode=='LD' else '3','-rc-lookahead','0','-forced-idr','1','-b_ref_mode','disabled' if mode=='LD' else 'middle','-zerolatency','1' if mode=='LD' else '0',str(bits)]
                        es=run(enc,d/'encode.log')
                        ds=run([FF,'-nostdin','-y','-v','warning','-i',str(bits),'-pix_fmt','yuv420p','-f','rawvideo',str(dec)],d/'decode.log')
                    if dec.stat().st_size!=framebytes*a.chunk: raise ValueError('Decoded frame count mismatch')
                    metric=[FF,'-nostdin','-v','info','-f','rawvideo','-pix_fmt','yuv420p','-s',f'{w}x{h}','-r',s['fps'],'-i',str(raw),'-f','rawvideo','-pix_fmt','yuv420p','-s',f'{w}x{h}','-r',s['fps'],'-i',str(dec),'-filter_complex',f'[0:v]split=2[rp][rs];[1:v]split=2[dp][ds];[rp][dp]psnr=stats_file={d}/psnr.log[p];[rs][ds]ssim=stats_file={d}/ssim.log[s]','-map','[p]','-map','[s]','-f','null','-']
                    run(metric,d/'metrics.log'); txt=(d/'metrics.log').read_text()
                    ps=re.search(r'PSNR y:(\S+) u:(\S+) v:(\S+) average:(\S+)',txt); ss=re.search(r'SSIM Y:(\S+).*All:(\S+)',txt)
                    if not ps or not ss: raise ValueError('Metrics missing')
                    row={key:s[key] for key in ['dataset','group','sequence','fps','fps_basis']}
                    row.update(codec=codec,mode=mode,qp=qp,chunk=k,first_frame=k*a.chunk,frames=a.chunk,width=w,height=h,pix_fmt='yuv420p',bytes=bits.stat().st_size,bpp=bits.stat().st_size*8/(w*h*a.chunk),enc_seconds=es,dec_seconds=ds,enc_fps=a.chunk/es,dec_fps=a.chunk/ds,psnr_y=float(ps[1]),psnr_u=float(ps[2]),psnr_v=float(ps[3]),psnr_avg=float(ps[4]),ssim_all=float(ss[2]))
                    (d/'result.json').write_text(json.dumps(row,indent=2)); dec.unlink()
                    export_results()
                    state['done']+=1; timings[(codec,mode)].append((time.monotonic()-jobstart)/(w*h*a.chunk)); progress()
                except Exception as e:
                    failures.append({'job':state['current'],'error':str(e)}); log(f'FAILED {state["current"]}: {e}')
                    (a.out/'failures.json').write_text(json.dumps(failures,indent=2))
            for file in work.iterdir(): file.unlink()
            work.rmdir()
    finally:
        stop.set(); export_results()
        log(f'FINISHED completed={state["done"]}/{state["total"]} failed={len(failures)} blocked_codecs={blocked}')
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
