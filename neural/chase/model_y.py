"""Original procedural illustration of the refreshed Model Y, in metres.

The nominal body envelope and wheelbase follow Tesla's Model Y dimensions.
The surface sections, glass outlines, lights and paint are hand drawn.
"""
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from scene import H, W, project

SCALE = 3


def screen(points):
    x, z, height = np.moveaxis(np.asarray(points), -1, 0)
    camera = np.stack([x, 1.65 - height, z - .55], axis=-1)
    u, v, depth = project(camera)
    return np.rint(SCALE * np.stack([u, v], axis=-1)).astype(np.int32), depth


def model_y():
    # Each station specifies distance from the body centre, half-width,
    # shoulder height, crown height, and the relative width of the upper cabin.
    stations = np.array([
        [-2.395, .68, .77, .86, .86], [-2.34, .81, .88, .95, .88],
        [-2.16, .90, .98, 1.08, .87], [-1.88, .95, 1.03, 1.15, .88],
        [-1.45, .96, 1.04, 1.36, .84], [-.88, .95, 1.04, 1.54, .79],
        [-.33, .94, 1.02, 1.62, .76], [.25, .94, 1.00, 1.60, .76],
        [.76, .95, .96, 1.48, .81], [1.24, .96, .90, 1.16, .87],
        [1.54, .94, .84, .96, .89], [1.98, .89, .75, .85, .88],
        [2.31, .77, .66, .75, .87], [2.395, .62, .60, .65, .85],
    ])
    spline = PchipInterpolator(stations[:, 0], stations[:, 1:], axis=0)
    faces = []

    def face(points, color):
        xy, depth = screen(points)
        faces.append((float(np.mean(depth)), xy, tuple(np.clip(color, 0, 255).astype(int).tolist())))

    def roof(z, t):
        w, shoulder, crown, cabin = np.moveaxis(spline(z), -1, 0)
        u = np.abs(t)
        outer = np.clip((u - cabin) / (1 - cabin), 0, 1)
        height = crown - .045 * np.minimum(u / cabin, 1)**2
        height -= (crown - .045 - shoulder) * outer**.85
        return np.stack([t*w, z, height], axis=-1)

    # Sample the shell in the car's metric coordinates. The glass borders follow
    # the curvature of the hatch and pillars rather than a screen-space overlay.
    z, t = np.meshgrid(np.linspace(-2.395, 2.395, 161), np.linspace(-1, 1, 121), indexing='ij')
    shell = roof(z, t)
    zc = (z[:-1, :-1] + z[1:, 1:]) / 2
    tc = (t[:-1, :-1] + t[1:, 1:]) / 2
    lateral = np.abs(tc)
    normals = np.cross(shell[:-1, 1:] - shell[:-1, :-1], shell[1:, :-1] - shell[:-1, :-1])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)
    illumination = np.clip(normals @ np.array([-.48, -.29, .83]), 0, 1)
    paint = np.array([225., 227., 225.]) * (.64 + .40*illumination[..., None])
    glazing = ((zc > -1.92 + .40*lateral**3) & (zc < 1.38 - .23*lateral**2)
               & (lateral < .77 + .035*np.cos(zc*1.6)))
    # Pearl-white roof rails frame one dark panoramic canopy. A thin divider
    # separates rear glass and windshield from the fixed glass roof.
    glass_light = 12*np.cos(zc*2.2) + 16*illumination + 17*(1-lateral) + 11*np.exp(-((tc+.43+zc*.18)/.19)**2)
    glass = np.array([23., 37., 44.]) + glass_light[..., None]
    divider = (np.abs(zc + .92 - .08*tc**2) < .032) | (np.abs(zc - .62 + .05*tc**2) < .031)
    glass[divider] *= .65
    colors = np.where(glazing[..., None], glass, paint)
    side_windows = ((lateral > .85) & (lateral < .98) & (zc > -1.53 + (lateral-.85)*3)
                    & (zc < 1.18 - (lateral-.85)*2.2))
    colors[side_windows] = np.array([33, 43, 47])
    for i, j in np.ndindex(colors.shape[:2]):
        face([shell[i,j], shell[i+1,j], shell[i+1,j+1], shell[i,j+1]], colors[i,j])

    # Lower doors and shoulder roll, then the front and rear fascia.
    levels = [(1., 0., [180,188,191]), (1., .13, [170,178,182]),
              (.98,.37,[145,154,160]), (.94,.56,[81,89,95]), (.88,.65,[47,53,59])]
    for end in (-1, 1):
        for i in range(0,z.shape[0]-1,2):
            z0,z1 = z[i,0],z[i+2,0]
            w0,h0,_,_ = spline(z0)
            w1,h1,_,_ = spline(z1)
            for (a,drop,col),(b,lower,_) in zip(levels, levels[1:]):
                face([(end*w0*a,z0,max(.19,h0-drop)),(end*w1*a,z1,max(.19,h1-drop)),
                      (end*w1*b,z1,max(.17,h1-lower)),(end*w0*b,z0,max(.17,h0-lower))],col)
        for axle in (-1.384,1.506):
            for r,x,col in [(.356,.941,[28,30,32]),(.239,.948,[77,84,89])]:
                theta=np.linspace(0,2*np.pi,32,endpoint=False)
                face([(end*x,axle+r*np.cos(a),.356+r*np.sin(a)) for a in theta],col)
        face([(end*.89,.65,1.04),(end*1.06,.55,1.05),(end*1.04,.38,1.06),
              (end*.91,.40,1.10)], [205,210,214])
    def tail(x, h, lift=0):
        # Rearward offsets and bevels round the corners of the hatch.
        return (x,-2.394 + .24*(abs(x)/.92)**3 + .09*(h-.74)**2 - lift,h)
    strips=[(.23,.69,[43,47,49]),(.32,.82,[57,62,65]),(.40,.87,[120,128,132]),
            (.52,.90,[174,181,185]),(.68,.91,[200,204,205]),(.80,.90,[211,214,214]),
            (.91,.85,[190,198,202]),(.99,.76,[159,170,177])]
    for (h0,w0,_),(h1,w1,col) in zip(strips,strips[1:]):
        for a,b in zip(np.linspace(-1,1,25)[:-1],np.linspace(-1,1,25)[1:]):
            face([tail(a*w0,h0),tail(b*w0,h0),tail(b*w1,h1),tail(a*w1,h1)],col)

    rgba=np.zeros((H*SCALE,W*SCALE,4),np.uint8)
    shadow,_=screen([(-.93,-2.37,.01),(.93,-2.37,.01),(.93,2.32,.01),(-.93,2.32,.01)])
    cv2.fillConvexPoly(rgba,shadow,(21,24,27,112))
    rgba=cv2.GaussianBlur(rgba,(27,27),0)
    for _,xy,color in sorted(faces,key=lambda f:-f[0]):
        cv2.fillPoly(rgba,[xy],color+(255,))

    # Camera always trails the vehicle; hatch trim is visible in this view.
    def trim(xs,hs,thickness,color):
        pts,_=screen([tail(x,h,.016) for x,h in zip(xs,hs)])
        cv2.polylines(rgba,[pts],False,color+(255,),thickness,cv2.LINE_AA)
    xs=np.linspace(-.88,.88,55)
    heights=.865+.058*(np.abs(xs)/.88)**2
    trim(xs,heights,8,(51,54,57))
    trim(xs,heights-.016,4,(171,43,46))
    trim(xs,heights-.021,2,(243,103,92))
    for sign in [-1,1]:
        trim([sign*.72,sign*.84,sign*.88],[.91,.92,.88],6,(197,47,46))
    plate,_=screen([tail(-.175,.61,.02),tail(.175,.61,.02),tail(.175,.49,.02),tail(-.175,.49,.02)])
    cv2.fillPoly(rgba,[plate],(67,71,74,255),lineType=cv2.LINE_AA)
    for i,letter in enumerate("TESLA"):
        pos,_=screen([tail((i-2)*.081-.024,.747,.022)])
        cv2.putText(rgba,letter,tuple(pos[0]),cv2.FONT_HERSHEY_SIMPLEX,.18,(54,59,64,255),1,cv2.LINE_AA)
    return cv2.resize(rgba,(W,H),interpolation=cv2.INTER_AREA)
