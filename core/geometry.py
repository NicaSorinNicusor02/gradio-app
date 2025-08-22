import json, numpy as np, cv2 as cv
from pyproj import CRS, Transformer

def yaw_level(img,yaw_deg,yaw_ref):
    if yaw_deg is None or yaw_ref is None: return img
    a=yaw_deg-yaw_ref; c=(img.shape[1]//2,img.shape[0]//2)
    M=cv.getRotationMatrix2D(c,a,1.0)
    return cv.warpAffine(img,M,(img.shape[1],img.shape[0]),flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT)

def rescale_to_gsd(img,gsd_cm,target):
    if gsd_cm is None or target is None: return img,1.0
    s=float(np.clip(gsd_cm/target,0.5,1.8))
    out=cv.resize(img,None,fx=s,fy=s,interpolation=cv.INTER_AREA if s<1 else cv.INTER_CUBIC)
    return out,s

def robust_H(p0,p1):
    if len(p0)<6 or len(p1)<6: return None
    H,_=cv.findHomography(p0,p1,method=cv.USAC_MAGSAC,ransacReprojThreshold=2.0,confidence=0.9999,maxIters=10000)
    return H

def angle_from_yaw(exifs):
    y=[e.get('yaw_deg') for e in exifs if e.get('yaw_deg') is not None]
    if not y: return 0.0
    a=np.rad2deg(np.arctan2(np.mean(np.sin(np.deg2rad(y))),np.mean(np.cos(np.deg2rad(y)))))
    return float(a)

def rotate_affine(img,angle_ccw):
    if abs(angle_ccw) < 1e-8:
        return img, np.eye(3, dtype=np.float64)[:2, :]
    h,w=img.shape[:2]; c=(w/2.0,h/2.0)
    M=cv.getRotationMatrix2D(c,angle_ccw,1.0)
    cs,sn=abs(M[0,0]),abs(M[0,1])
    nw=int(w*cs+h*sn); nh=int(w*sn+h*cs)
    M[0,2]+=(nw/2.0)-c[0]; M[1,2]+=(nh/2.0)-c[1]
    out=cv.warpAffine(img,M,(nw,nh),flags=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT,borderValue=(0,0,0))
    return out,M

def lon_to_epsg(lon,lat):
    z=int((lon+180)//6)+1
    return 32600+z if lat>=0 else 32700+z

def to_utm_xy(lats,lons):
    lat0=float(np.nanmean(lats)); lon0=float(np.nanmean(lons))
    epsg=lon_to_epsg(lon0,lat0)
    tf=Transformer.from_crs(CRS.from_epsg(4326),CRS.from_epsg(epsg),always_xy=True)
    X=[]; Y=[]
    for lo,la in zip(lons,lats):
        if lo is None or la is None: X.append(np.nan); Y.append(np.nan); continue
        x,y=tf.transform(lo,la); X.append(x); Y.append(y)
    return np.array(X,np.float64),np.array(Y,np.float64),int(epsg)

def estimate_geo(Hs_shift,exifs):
    pix=[]; en=[]
    for H,e in zip(Hs_shift,exifs):
        if e.get('w',0)<=0 or e.get('h',0)<=0: continue
        c=np.array([e['w']*0.5,e['h']*0.5,1.0],np.float64)
        p=H@c
        if not np.isfinite(p[2]) or p[2]==0: continue
        u,v=(p[:2]/p[2])
        if e.get('lat') is None or e.get('lon') is None: continue
        pix.append([u,v]); en.append([e['lon'],e['lat']])
    if len(pix)<4: return None,None,None,None
    lats=np.array([q[1] for q in en],np.float64); lons=np.array([q[0] for q in en],np.float64)
    X,Y,epsg=to_utm_xy(lats,lons); m=np.isfinite(X)&np.isfinite(Y); P=np.array(pix,np.float64)[m]; Q=np.stack([X[m],Y[m]],1)
    A,_=cv.estimateAffinePartial2D(P,Q,method=cv.RANSAC,ransacReprojThreshold=1.5,confidence=0.999)
    if A is None: return None,None,None,None
    th=np.degrees(np.arctan2(A[1,0],A[0,0])); sc=float(np.hypot(A[0,0],A[1,0]))
    return A,float(th),sc,int(epsg)

def apply_hflip(img,A_world):
    h,w=img.shape[:2]; F=np.array([[-1,0,w-1],[0,1,0],[0,0,1]],np.float64)
    out=cv.flip(img,1)
    A3=np.eye(3); A3[:2,:]=A_world
    A_out=(A3@np.linalg.inv(F))[:2,:]
    return out,A_out,F
