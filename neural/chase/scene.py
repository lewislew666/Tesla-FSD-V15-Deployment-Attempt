"""Causal KITTI stereo reprojection into a camera rigidly attached behind the car."""
import io
import sys
from collections import deque
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"neural"), str(ROOT/"baselines/stereo")]
from reality import SharedVision
from kitti import calibration, DRIVE
from demo import stereo

W,H = 768,432
F,CX,CY = 516., (W-1)/2, 207.
PITCH = np.deg2rad(27.)
C,S = np.cos(PITCH),np.sin(PITCH)
# Virtual camera position and axes in the left rectified optical frame.
EYE = np.array([0.,-4.7,-9.2],np.float32)
AXES = np.array([[1,0,0],[0,C,S],[0,-S,C]],np.float32)
V,U = np.indices((H,W),dtype=np.float32)
RAYS = np.stack([(U-CX)/F,(V-CY)/F,np.ones_like(U)],-1) @ AXES.T


def affine(r,t):
    a=np.eye(4)
    a[:3,:3],a[:3,3]=r,t
    return a


def local_pose(row,scale):
    lat,lon,alt,roll,pitch,yaw=row[:6]
    x=scale*6378137*np.deg2rad(lon)
    y=scale*6378137*np.log(np.tan(np.pi/4+np.deg2rad(lat)/2))
    def rot(axis,angle):
        v=np.zeros(3); v[axis]=angle
        return cv2.Rodrigues(v)[0]
    return affine(rot(2,yaw)@rot(1,pitch)@rot(0,roll),[x,y,alt])


def project(p):
    q=(p-EYE) @ AXES
    z=q[...,2]
    return F*q[...,0]/np.maximum(.1,z)+CX, F*q[...,1]/np.maximum(.1,z)+CY, z


def apply(p,a):
    return np.einsum("...j,ij->...i",p,a[:3,:3],optimize=False)+a[:3,3]


def read_camera(z,cam,i):
    return np.asarray(Image.open(io.BytesIO(z.read(f"{DRIVE}/{cam}/data/{i:010d}.png"))).convert("RGB"))


def generate(args,tmp):
    tmp.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.calib) as z:
        cam=calibration(z,"calib_cam_to_cam.txt")
        velo=calibration(z,"calib_velo_to_cam.txt")
        imu=calibration(z,"calib_imu_to_velo.txt")
    p2,p3=[cam[k].reshape(3,4) for k in ("P_rect_02","P_rect_03")]
    fx,fy,cx,cy=p2[0,0],p2[1,1],p2[0,2],p2[1,2]
    rect=affine(cam["R_rect_00"].reshape(3,3),np.linalg.solve(p2[:,:3],p2[:,3]))
    trans=lambda d:affine(d["R"].reshape(3,3),d["T"])
    cam_to_imu=np.linalg.inv(rect@trans(velo)@trans(imu))
    model=SharedVision.from_checkpoint(str(args.checkpoint),meta_repo=args.meta_repo).eval()
    history=deque(maxlen=12)
    with ZipFile(args.drive) as z, torch.inference_mode():
        times=z.read(f"{DRIVE}/image_02/timestamps.txt").decode().splitlines()
        n=len(times)
        rows=[np.fromstring(z.read(f"{DRIVE}/oxts/data/{i:010d}.txt").decode(),sep=" ") for i in range(n)]
        scale=np.cos(np.deg2rad(rows[0][0]))
        poses=np.array([local_pose(row,scale)@cam_to_imu for row in rows])
        poses=np.linalg.inv(poses[0])[None]@poses
        np.savez(tmp/"time_pose.npz",stamps=np.array(times),poses=poses)
        previous=None
        for i in range(n):
            left,right=[read_camera(z,camid,i) for camid in ("image_02","image_03")]
            dep=stereo(left,right,p2[0,3]-p3[0,3],85.,128)
            im=torch.from_numpy(cv2.resize(left,(448,160),interpolation=cv2.INTER_AREA))[None].permute(0,3,1,2).float()/255
            tokens=model.patch_grid(im)[0].permute(1,2,0)
            if i==0:
                sample=tokens.reshape(-1,tokens.shape[-1])
                mean=sample.mean(0)
                _,_,vt=torch.linalg.svd(sample-mean,full_matrices=False)
                axes=vt[:3].T
                s=(sample-mean)@axes
                lo,hi=torch.quantile(s,torch.tensor([.02,.98]),dim=0)
            code=(((tokens-mean)@axes-lo)/(hi-lo)).clamp(0,1).mul(255).byte().numpy()
            yy,xx=np.nonzero(np.isfinite(dep[::2,::2])); yy*=2; xx*=2
            dz=dep[yy,xx].astype(np.float32)
            local=np.stack(((xx-cx)*dz/fx,(yy-cy)*dz/fy,dz),1).astype(np.float32)
            world=apply(local,poses[i].astype(np.float32))
            codepts=code[yy*code.shape[0]//left.shape[0],xx*code.shape[1]//left.shape[1]]
            entry=(world,left[yy,xx],codepts,i,left)
            inv=np.linalg.inv(poses[i]).astype(np.float32)
            depth=np.full(H*W,np.inf,np.float32)
            surf=np.zeros((H*W,3),np.uint8); feat=np.zeros_like(surf)
            for pts,rgb,lat,j,_ in list(history)+[entry]:
                xyz=apply(pts,inv)
                u,v,d=project(xyz)
                u,v=np.rint(u).astype(int),np.rint(v).astype(int)
                ok=(d>1)&(u>=0)&(v>=0)&(u<W)&(v<H)
                if j!=i:
                    # Reject old foreground observations that the current stereo rays contradict.
                    sx=np.rint(fx*xyz[:,0]/np.maximum(.1,xyz[:,2])+cx).astype(int)
                    sy=np.rint(fy*xyz[:,1]/np.maximum(.1,xyz[:,2])+cy).astype(int)
                    inside=(sx>=0)&(sy>=0)&(sx<left.shape[1])&(sy<left.shape[0])&(xyz[:,2]>0)
                    indices=np.flatnonzero(inside & ok)
                    new=dep[sy[indices],sx[indices]]
                    ok[indices[new-xyz[indices,2]>1.0]]=False
                ids=np.flatnonzero(ok)
                pix=v[ids]*W+u[ids]
                np.minimum.at(depth,pix,d[ids])
                use=d[ids]<=depth[pix]+.04
                where=pix[use]; surf[where]=rgb[ids[use]]; feat[where]=lat[ids[use]]
            depth=depth.reshape(H,W); surf=surf.reshape(H,W,3); feat=feat.reshape(H,W,3)
            measured=np.isfinite(depth)
            # A far plane supplies distant color; ground intersection guides disoccluded road.
            ray_z=np.minimum(130.,(1.65-EYE[1])/np.maximum(RAYS[...,1],.001))
            approx=EYE+ray_z[...,None]*RAYS
            sx=(fx*approx[...,0]/np.maximum(.1,approx[...,2])+cx).astype(np.float32)
            sy=(fy*approx[...,1]/np.maximum(.1,approx[...,2])+cy).astype(np.float32)
            guide=cv2.remap(left,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            if measured.any():
                from scipy.ndimage import distance_transform_edt
                distance,nearest=distance_transform_edt(~measured,return_indices=True)
                nv,nu=nearest
                nearest_rgb=surf[nv,nu]
                near=(distance<20)|((approx[...,2]<4)|(sy>=left.shape[0]))
                guide[near]=nearest_rgb[near]
                feat=feat[nv,nu]
            # Inverse-sample the original images at the fused surface. This avoids
            # point-splat aliasing and copies photograph detail from visible road.
            dense_depth=np.where(distance<9,depth[nv,nu],ray_z).astype(np.float32)
            dense_depth=cv2.medianBlur(dense_depth,5)
            coords=EYE+RAYS*dense_depth[...,None]
            total=np.zeros((H,W,3),np.float32)
            denom=np.zeros((H,W),np.float32)
            for _,_,_,j,photo in list(history)[::3]+[entry]:
                local=apply(coords,(np.linalg.inv(poses[j])@poses[i]).astype(np.float32))
                zcam=np.maximum(.1,local[...,2])
                mx=(fx*local[...,0]/zcam+cx).astype(np.float32)
                my=(fy*local[...,1]/zcam+cy).astype(np.float32)
                edge=np.minimum.reduce([mx,my,left.shape[1]-1-mx,left.shape[0]-1-my])
                weight=np.clip(edge/16,0,1)*(zcam>2)*np.exp(-abs(i-j)/7)
                if j==i:
                    surfz=cv2.remap(np.nan_to_num(dep,nan=0.).astype(np.float32),mx,my,cv2.INTER_NEAREST)
                    weight*=((surfz==0)|(surfz>zcam-1))
                color=cv2.remap(photo,mx,my,cv2.INTER_LINEAR)
                total+=color*weight[...,None]
                denom+=weight
            valid=denom>.01
            guide[valid]=np.clip(total[valid]/denom[valid,None],0,255)
            # Anchor directly supported near surfaces; historical reprojection guides holes.
            guide[measured]=(.5*guide[measured]+.5*surf[measured]).astype(np.uint8)
            if previous is None:
                uv=np.stack([U,V],-1)
            else:
                back=apply(coords,(np.linalg.inv(previous)@poses[i]).astype(np.float32))
                bx,by,bz=project(back); uv=np.stack((bx,by),-1)
                uv[bz<.1]=-100
            np.savez(tmp/f"{i:03d}.npz",guide=guide,measured=measured,feat=feat,
                     flowmap=np.clip(uv,-1500,1500).astype(np.float16),front=cv2.resize(left,(448,136)))
            previous=poses[i]
            if i%2==0: history.append(entry)
            if i%20==0: print("projected frame",i,flush=True)
    return times
