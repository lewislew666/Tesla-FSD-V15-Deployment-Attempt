"""Procedural sedan, drawn in the same metric camera as the road."""
import cv2
import numpy as np
from scene import project,W,H


def vehicle():
    ground=1.65
    def point(x,z,height):
        return np.array([x,ground-height,z-.55])
    faces=[]
    def poly(vertices,color):
        points=np.array([point(*v) for v in vertices])
        u,v,d=project(points)
        pts=np.rint(2*np.stack([u,v],1)).astype(np.int32)
        faces.append((d.mean(),pts,tuple(map(int,color))))
    # Paint shell, bumper shoulders and hood formed by connected cross sections.
    sections=[(-2.25,.68,.56),(-2.14,.84,.84),(-1.85,.92,.93),(-1.35,.945,1.00),
              (-.8,.95,1.02),(.5,.94,1.01),(1.15,.92,.93),(1.75,.87,.82),(2.16,.73,.71)]
    def body(z,w,h):
        return [(-w*.88,z,.34),(-w,z,.53),(-w,z,h-.10),(-w*.86,z,h),
                (0,z,h+.035),(w*.86,z,h),(w,z,h-.10),(w,z,.53),(w*.88,z,.34)]
    for a,b in zip(sections,sections[1:]):
        s,t=body(*a),body(*b)
        for k,col in enumerate([(46,62,73),(75,96,110),(145,169,177),(181,196,200),
                                (184,200,204),(138,160,172),(65,86,100),(46,60,73)]):
            poly([s[k],s[k+1],t[k+1],t[k]],col)
    poly(body(*sections[0]),(65,82,94)); poly(body(*sections[-1]),(135,151,153))
    # Cabin surfaces: rear glass, roof, windshield and side glazing.
    rings=[(-1.65,.76,.96),(-.88,.67,1.46),(-.53,.65,1.52),(.43,.64,1.51),(.64,.65,1.45),(1.21,.77,.99)]
    def cabin(z,w,h):
        return [(-w,z,h),(-w*.6,z,h+.035),(w*.6,z,h+.035),(w,z,h)]
    for i,(a,b) in enumerate(zip(rings,rings[1:])):
        s,t=cabin(*a),cabin(*b)
        for j in range(3):
            color=([(49,68,80),(55,76,88),(46,64,77)] if i in [0,4] else
                   [(163,182,190),(194,207,210),(170,189,198)])[j]
            poly([s[j],s[j+1],t[j+1],t[j]],color)
        for sign in [-1,1]:
            poly([(sign*a[1],a[0],a[2]),(sign*b[1],b[0],b[2]),
                  (sign*.84,b[0],1.),(sign*.84,a[0],1.)],(38,55,65))
    for sign in [-1,1]:
        poly([(sign*.66,-.52,1.5),(sign*.66,-.43,1.5),(sign*.84,-.38,1.0),(sign*.84,-.49,1.0)],(98,115,124))
        poly([(sign*.84,.65,1.12),(sign*1.03,.56,1.10),(sign*1.04,.41,1.07),(sign*.84,.43,1.07)],(142,164,177))
        for z in [-1.47,1.35]:
            for rad,x,col in [(.34,.93,(32,37,42)),(.21,.945,(111,125,133)),(.10,.95,(57,71,79))]:
                poly([(sign*x,z+rad*np.cos(t),.35+rad*np.sin(t)) for t in np.arange(20)*np.pi/10],col)
        poly([(sign*.81,-2.15,.80),(sign*.40,-2.24,.78),(sign*.39,-2.24,.69),(sign*.81,-2.16,.69)],(161,36,45))
        poly([(sign*.74,-2.17,.785),(sign*.45,-2.24,.766),(sign*.44,-2.24,.745),(sign*.74,-2.17,.759)],(246,95,82))
    poly([(-.20,-2.255,.62),(.20,-2.255,.62),(.20,-2.255,.50),(-.20,-2.255,.50)],(210,215,206))
    rgba=np.zeros((H*2,W*2,4),np.uint8)
    corners=np.array([point(x,z,.02) for x,z in [(-1.01,-2.32),(1.01,-2.32),(1.01,2.25),(-1.01,2.25)]])
    u,v,_=project(corners)
    cv2.fillConvexPoly(rgba,np.rint(np.stack([u,v],1)*2).astype('int32'),(10,16,22,110))
    rgba=cv2.GaussianBlur(rgba,(17,17),0)
    for _,pts,rgb in sorted(faces,key=lambda f:-f[0]):
        cv2.fillPoly(rgba,[pts],rgb+(255,),lineType=cv2.LINE_AA)
    return cv2.resize(rgba,(W,H),interpolation=cv2.INTER_AREA)


def finish(rgba,front):
    """Modulate glazing and clear coat with low-frequency current-scene reflections."""
    paint=rgba[...,:3].astype(np.float32)
    y,x=np.nonzero(rgba[...,3]>235)
    region=paint[y.min():y.max()+1,x.min():x.max()+1]
    h,w=region.shape[:2]
    reflection=cv2.resize(front[:max(1,front.shape[0]//2)],(w,h),interpolation=cv2.INTER_AREA)[::-1].astype(np.float32)
    reflection=cv2.GaussianBlur(reflection,(0,0),2.3)
    lum=region.mean(-1)
    glass=((lum>45)&(lum<87))[:,:,None]
    metallic=(lum>125)[:,:,None]
    shine=np.exp(-((np.linspace(-1,1,w)-.28)/.25)**2)[None,:,None]*10
    region[:]=np.clip(np.where(glass,region*.75+reflection*.25,region)+metallic*((reflection-110)*.075+shine),0,255)
    return paint
