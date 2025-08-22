import re, numpy as np, exifread
from PIL import Image

def _rf(x):
    try: return float(x.num)/float(x.den)
    except:
        try: return float(x)
        except: return None

def _dms(vals, ref):
    try:
        d=_rf(vals[0]); m=_rf(vals[1]); s=_rf(vals[2])
        v=d+m/60.0+s/3600.0
        if str(ref).upper().startswith(('S','W')): v=-v
        return float(v)
    except: return None

def _extract_xmp_yaw_bytes(b):
    try:
        i = b.find(b"<x:xmpmeta")
        if i < 0:
            return None, None
        j = b.find(b"</x:xmpmeta>", i)
        if j < 0:
            return None, None
        s = b[i:j+12].decode("utf-8", "ignore")

        import re
        vals = {}

        # Attribute form:  <... FlightYawDegree="-139.70" ...>
        for m in re.finditer(r'(?:[\w\-]+:)?(FlightYawDegree|GimbalYawDegree)="([+\-]?\d+(?:\.\d+)?)"', s):
            vals[m.group(1)] = float(m.group(2))

        # Element form:  <drone-dji:FlightYawDegree>-139.70</drone-dji:FlightYawDegree>
        for m in re.finditer(r'<(?:[\w\-]+:)?(FlightYawDegree|GimbalYawDegree)>\s*([+\-]?\d+(?:\.\d+)?)\s*</(?:[\w\-]+:)?\1>', s):
            if m.group(1) not in vals:
                vals[m.group(1)] = float(m.group(2))

        fy = vals.get("FlightYawDegree")
        gy = vals.get("GimbalYawDegree")
        return fy, gy
    except:
        return None, None

def _orient_deg(tags):
    t=tags.get('Image Orientation')
    if t is None: return 0.0
    try:
        v=int(str(t).split()[0])
    except:
        try: v=int(str(t))
        except: return 0.0
    if v==1: return 0.0
    if v==3: return 180.0
    if v==6: return -90.0
    if v==8: return 90.0
    return 0.0

def read_meta(path):
    d={'lat':0.0,'lon':0.0,'yaw_deg':0.0,'alt_m':0.0,'w':0,'h':0,'focal_mm':0.0,'focal35mm':0.0}
    with open(path,'rb') as f:
        raw=f.read(); f.seek(0); tags=exifread.process_file(f,details=False)

    def get(n):
        t=tags.get(n)
        if t is None: return None
        try:
            v=getattr(t,'values',t)
            if isinstance(v,(list,tuple)) and v: return _rf(v[0])
            return _rf(v)
        except:
            try: return float(str(t))
            except: return None

    v=get('EXIF FocalLength'); d['focal_mm']=v if v is not None else 0.0
    v=get('EXIF FocalLengthIn35mmFilm'); d['focal35mm']=v if v is not None else 0.0
    v=get('GPS GPSAltitude'); d['alt_m']=v if v is not None else 0.0

    latv=tags.get('GPS GPSLatitude'); latr=tags.get('GPS GPSLatitudeRef')
    lonv=tags.get('GPS GPSLongitude'); lonr=tags.get('GPS GPSLongitudeRef')
    if latv and lonv and latr and lonr:
        lat=_dms(getattr(latv,'values',[]),latr); lon=_dms(getattr(lonv,'values',[]),lonr)
        d['lat']=lat if lat is not None else 0.0
        d['lon']=lon if lon is not None else 0.0

    fy,gy=_extract_xmp_yaw_bytes(raw)
    yaw=fy if fy is not None else (gy if gy is not None else None)
    if yaw is None:
        for k in ['XMP FlightYawDegree','XMP GimbalYawDegree','MakerNote Yaw','Image Tag 0x0011']:
            v=get(k)
            if v is not None:
                yaw=v; break

    rot=_orient_deg(tags)
    if yaw is None: yaw=0.0
    d['yaw_deg']=float(yaw - rot)

    with Image.open(path) as im:
        d['w'],d['h']=im.size
    for k in ['lat','lon','alt_m','focal_mm','focal35mm','yaw_deg']:
        if d.get(k) is None: d[k]=0.0
    return d

def sensor_width_mm(f_mm,f35): return 36.0*f_mm/f35 if (f_mm>0 and f35>0) else None
def intrinsics(e):
    sw=sensor_width_mm(e['focal_mm'],e['focal35mm'])
    if sw is None or e['focal_mm']<=0 or e['w']<=0: return None,None
    fx=e['focal_mm']/sw*e['w']; cx=(e['w']-1)*0.5; cy=(e['h']-1)*0.5
    K=np.array([[fx,0,cx],[0,fx,cy],[0,0,1]],np.float64); D=np.zeros((5,1),np.float64)
    return K,D
def gsd_cm_per_px(alt_m,focal_mm,sensor_w_mm,img_w):
    if alt_m is None or alt_m<=0 or focal_mm<=0 or sensor_w_mm is None or img_w<=0: return None
    g=(alt_m*(sensor_w_mm*1e-3))/(focal_mm*1e-3*img_w); return g*100.0
