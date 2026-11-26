"""Masked conditional RGB diffusion. All training targets come from earlier road frames."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIZE=(384,216)
STEPS=(.94,.76,.54,.32,.12)


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).float()[None]/127.5-1


def forward_sample(clean,noise,t):
    a=torch.cos(t*np.pi/2)
    b=torch.sin(t*np.pi/2)
    return a*clean+b*noise


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i,o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU(),
                                 nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU())
        self.high=block(14,20)
        self.low=block(20,40)
        self.mid=block(40,48)
        self.up=block(88,32)
        self.end=nn.Sequential(block(52,20),nn.Conv2d(20,3,3,padding=1))

    def forward(self,x,t,guide,mask,sem,prev):
        cond=torch.cat([x,guide,mask,sem,prev,t.expand_as(mask)],1)
        h=self.high(cond)
        l=self.low(F.avg_pool2d(h,2))
        m=self.mid(F.avg_pool2d(l,2))
        up=self.up(torch.cat([l,F.interpolate(m,size=l.shape[-2:],mode="bilinear",align_corners=False)],1))
        return (guide+self.end(torch.cat([h,F.interpolate(up,size=h.shape[-2:],mode="bilinear",align_corners=False)],1))).clamp(-1,1)


def small(row):
    resize=lambda im:cv2.resize(im,SIZE,interpolation=cv2.INTER_AREA)
    return tensor(resize(row["guide"])), torch.from_numpy(resize(row["measured"].astype(np.float32)))[None,None],tensor(resize(row["feat"]))


def train(cache,steps=480):
    torch.manual_seed(61)
    rng=np.random.default_rng(61)
    frames=[small(np.load(cache/f"{i:03d}.npz")) for i in range(36)]
    model=Denoiser()
    opt=torch.optim.AdamW(model.parameters(),lr=.0007)
    for step in range(steps):
        patches=[]
        for _ in range(4):
            g,m,s=frames[rng.integers(len(frames))]
            y,x=rng.integers(0,216-72),rng.integers(0,384-96)
            patches.append([a[:,:,y:y+72,x:x+96] for a in (g,m,s)])
        target,measured,sem=[torch.cat(part) for part in zip(*patches)]
        choose=F.interpolate(torch.rand(4,1,7,9),size=target.shape[-2:],mode="bicubic",align_corners=False)
        known=(choose>.49).float()*measured
        smooth=F.avg_pool2d(F.pad(target,(4,4,4,4),mode='replicate'),9,stride=1)
        warped=torch.roll(smooth,(int(rng.integers(-7,8)),int(rng.integers(-7,8))),(-2,-1))
        prior=known*target+(1-known)*warped
        prev=.4*smooth+.6*torch.roll(target,(int(rng.integers(-5,6)),int(rng.integers(-5,6))),(-2,-1))
        t=torch.rand(4,1,1,1)*.97+.015
        noisy=forward_sample(target,torch.randn_like(target),t)
        out=model(noisy,t,prior,known,sem,prev)
        w=(.1+.9*measured)*(1.5-known)
        loss=(((out-target).square()+.12*(out-target).abs())*w).sum()/(3*w.sum())
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%120==0: print("diffusion training",step+1,round(float(loss.detach()),4),flush=True)
    return model.eval()


@torch.no_grad()
def sample(model,row,previous,rng):
    guide,mask,sem=small(row)
    raw=row["guide"]
    prev=guide
    if previous is not None:
        uv=row["flowmap"].astype(np.float32)
        repro=cv2.remap(previous,uv,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        stable=(uv[...,0]>=0)&(uv[...,0]<raw.shape[1]-1)&(uv[...,1]>=0)&(uv[...,1]<raw.shape[0]-1)
        repro[~stable]=raw[~stable]
        prev=tensor(cv2.resize(repro,SIZE,interpolation=cv2.INTER_AREA))
    noise=torch.randn(guide.shape,generator=rng)
    x=forward_sample(prev,noise,torch.tensor(STEPS[0]))
    for now,nxt in zip(STEPS,STEPS[1:]+(0.,)):
        t=torch.tensor(now).reshape(1,1,1,1)
        pred=model(x,t,guide,mask,sem,prev)
        x0=mask*guide+(1-mask)*pred
        eps=(x-torch.cos(t*np.pi/2)*x0)/torch.sin(t*np.pi/2)
        x=forward_sample(x0,eps,torch.tensor(nxt))
    low=((x[0].permute(1,2,0)+1)*127.5).clamp(0,255).numpy()
    low=cv2.resize(low,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_CUBIC)
    # Preserve image detail near observed surfaces. Distant unsupported regions use the denoised estimate.
    support=cv2.GaussianBlur(row["measured"].astype(np.float32),(0,0),2)[...,None]
    hi=raw.astype(np.float32)-cv2.GaussianBlur(raw.astype(np.float32),(0,0),1.15)
    return np.clip(low+hi*np.minimum(1.,2*support+.15),0,255).astype(np.uint8)
