"""Train/replay a five-step conditional diffusion chase view, with timestamped MP4 outputs."""
import argparse
import csv
from fractions import Fraction
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from scene import generate,H,W
from vehicle import vehicle, finish
from diffusion import train,sample,Denoiser,STEPS


def encode(folder,name,pts,duration,dest):
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    lines=["ffconcat version 1.0"]
    for i,dt in enumerate(duration):
        lines += [f"file '{name}-{i:03d}.png'", "option framerate 1000000",f"duration {dt/1e6:.6f}"]
    listing=folder/(name+".ffconcat")
    listing.write_text("\n".join(lines)+"\n")
    filt=f"setts=duration='if(eq(N,{len(pts)-1}),{duration[-1]},DURATION)'"
    subprocess.run([ff,"-hide_banner","-loglevel","error","-y","-safe","0","-f","concat","-i",str(listing),
                    "-fps_mode","vfr","-enc_time_base","1:1000000","-c:v","libx264","-bf","0","-crf","17",
                    "-pix_fmt","yuv420p","-video_track_timescale","1000000","-bsf:v",filt,"-movflags","+faststart",str(dest)],check=True)
    audit=subprocess.check_output([ff,"-v","error","-i",str(dest),"-map","0:v:0","-c:v","copy","-f","framecrc","-"]).decode()
    tb=Fraction(next(l.split()[-1] for l in audit.splitlines() if l.startswith("#tb")))
    packets=[l.split(",") for l in audit.splitlines() if l and not l.startswith("#")]
    actual=np.array([round(int(p[2])*tb*1000000) for p in packets])
    assert len(actual)==len(pts) and np.array_equal(actual,pts),(actual,pts)
    last_duration=round(int(packets[-1][3])*tb*1000000)
    assert last_duration==duration[-1],(last_duration,duration[-1])
    subprocess.run([ff,"-v","error","-xerror","-i",str(dest),"-f","null","-"],check=True)
    return actual


def font(size):
    try:
        return ImageFont.truetype("Arial.ttf",size)
    except OSError:
        return ImageFont.load_default(size=size)


def evidence_view(measured,alpha):
    evidence=np.where(measured[...,None],np.array([83,174,167]),np.array([197,146,82])).astype(np.uint8)
    evidence[alpha>.80]=[193,204,218]
    return Image.fromarray(evidence).resize((340,191),Image.Resampling.BOX)


def render(args,cache,model):
    record=np.load(cache/"time_pose.npz")
    stamps=record["stamps"]
    ns=np.array(stamps,dtype="datetime64[ns]").astype("int64")
    us=np.rint((ns-ns[args.start])/1000).astype(int)
    frames=list(range(args.start,len(stamps)))
    pts=us[frames]
    dt=np.diff(pts,append=pts[-1]+round(np.median(np.diff(pts))))
    ego=vehicle(args.vehicle)
    alpha=ego[...,3]/255.
    previous=None
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    temp=cache/"frames";temp.mkdir(exist_ok=True)
    rng=torch.Generator().manual_seed(722)
    rows=[]; perf=[]
    for k,i in enumerate(frames):
        with np.load(cache/f"{i:03d}.npz") as row:
            tic=time.perf_counter()
            road=sample(model,row,previous,rng)
            perf.append(time.perf_counter()-tic)
            previous=road
            view=np.rint((1-alpha[...,None])*road+alpha[...,None]*finish(ego,row["front"])).astype(np.uint8)
            image=Image.fromarray(view)
            image.save(temp/f"chase-{k:03d}.png")
            measured=row["measured"]
            b=Image.new("RGB",(1152,514),"#122027")
            draw=ImageDraw.Draw(b)
            draw.text((16,10),"CHASE VIEW",font=font(20),fill="#edf3f3")
            draw.text((792,12),"SYNCHRONIZED CAMERA",font=font(16),fill="#edf3f3")
            b.paste(image,(12,42))
            b.paste(Image.fromarray(row["front"]).resize((340,104),Image.Resampling.LANCZOS),(792,43))
            draw.text((792,175),"SOURCE COVERAGE",font=font(15),fill="#edf3f3")
            b.paste(evidence_view(measured,alpha),(792,199))
            for j,(label,color) in enumerate([("Stereo-supported", "#53aea7"),("Completed / approximate","#c59252"),("Illustrative ego vehicle","#c1ccda")]):
                y=400+j*23
                draw.rectangle((792,y+2,804,y+13),fill=color);draw.text((814,y),label,font=font(13),fill="#d1dfe1")
            draw.text((14,483),f"KITTI 0001    /    FRAME {i:03d}    /    {stamps[i].split()[1]}    /    +{pts[k]/1e6:.3f} s",font=font(14),fill="#cbdcdd")
            b.save(temp/f"sync-{k:03d}.png")
            if k==len(frames)//2: b.save(out/"preview.jpg",quality=94)
            rows.append([i,str(stamps[i]),int(pts[k]),int(dt[k]),round(float(measured[alpha<.01].mean()),5)])
    for tag in ["chase","sync"]:
        encode(temp,tag,pts,dt,out/(tag+".mp4"))
    with (out/"timing.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["source_frame","kitti_left_timestamp","mp4_pts_us","duration_us","stereo_supported_background_fraction"]);w.writerows(rows)
    stats=dict(vehicle=args.vehicle,frames=len(frames),start_frame=frames[0],end_frame=frames[-1],duration_s=(pts[-1]+dt[-1]).item()/1e6,
               max_pts_error_us=0,source_rounding_error_ns=int(abs((ns[frames]-ns[frames[0]])-1000*pts).max()),
               verification="Compared packet PTS and final duration in both exported MP4 files; fully decoded both.",
               last_duration="median inter-frame duration",mean_supported_background=float(np.mean([r[-1] for r in rows])),
               mean_diffusion_seconds_per_frame=float(np.mean(perf)),timing_scope="CPU diffusion and full-resolution composite preparation only; excludes geometry and video encoding",
               model_steps=len(STEPS),virtual_eye_left_camera_xyz_m=[0,-4.7,-9.2],pitch_degrees=27,
               fitted_frames=[0,35],rendered_size=[W,H],training_steps=480)
    (out/"metrics.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(stats,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("drive","calib","checkpoint","meta-repo","cache","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--vehicle",choices=("model-y","sedan"),default="model-y")
    p.add_argument("--reuse-cache",action="store_true",help="Use the precomputed geometry for this drive/rig.")
    p.add_argument("--train",action="store_true",help="Fit a new diffusion renderer on frames 0–35.")
    p.add_argument("--weights",type=Path,default=Path(__file__).parent/"result/denoiser.pth")
    p.add_argument("--start",type=int,default=40)
    args=p.parse_args()
    if not 36<=args.start<=105:
        p.error("start must be in [36,105] to keep fitting frames separate and establish frame duration")
    torch.set_num_threads(4)
    if not args.reuse_cache: generate(args,args.cache)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.train:
        model=train(args.cache)
        torch.save(model.state_dict(),args.out/"denoiser.pth")
    else:
        model=Denoiser().eval()
        model.load_state_dict(torch.load(args.weights,map_location="cpu",weights_only=True))
    render(args,args.cache,model)

if __name__=="__main__":
    main()
