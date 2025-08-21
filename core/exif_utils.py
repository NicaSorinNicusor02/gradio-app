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

_XMP_NUM=rb'([+\-]?\d+(?:\.\d+)?)'
def _read_dji_xmp(path):
    try:
        with open(path,'rb') as f: data=f.read()
        def grab(*names):
            for n in names:
                m=re.search(rb'[\w\-:]*'+n+rb'\s*=\s*"'+_XMP_NUM+rb'"',data)
                if m: return float(m.group(1).decode('ascii'))
            return None
        lat=grab(b'GpsLatitude'); lon=grab(b'GpsLongitude')
        gy=grab(b'GimbalYawDegree'); fy=grab(b'FlightYawDegree'); yaw=gy if gy is not None else fy
        ra=grab(b'RelativeAltitude'); aa=grab(b'AbsoluteAltitude'); alt=ra if ra is not None else aa
        return {'lat':lat,'lon':lon,'yaw_deg':yaw,'alt_m':alt}
    except: return {'lat':None,'lon':None,'yaw_deg':None,'alt_m':None}

def read_meta(path):
    d={'lat':None,'lon':None,'yaw_deg':None,'alt_m':0.0,'w':0,'h':0,'focal_mm':0.0,'focal35mm':0.0}
    with open(path,'rb') as f: tags=exifread.process_file(f,details=False)
    def get(n):
        t=tags.get(n); 
        if t is None: return None
        try:
            v=getattr(t,'values',t)
            if isinstance(v,(list,tuple)) and v: return _rf(v[0])
            return _rf(v)
        except:
            try: return float(str(t))
            except: return None
    d['focal_mm']=get('EXIF FocalLength') or 0.0
    d['focal35mm']=get('EXIF FocalLengthIn35mmFilm') or 0.0
    alt=get('GPS GPSAltitude'); d['alt_m']=alt if alt is not None else 0.0
    for k in ['XMP GimbalYawDegree','XMP FlightYawDegree','MakerNote Yaw','Image Tag 0x0011']:
        v=get(k)
        if v is not None: d['yaw_deg']=v; break
    latv=tags.get('GPS GPSLatitude'); latr=tags.get('GPS GPSLatitudeRef')
    lonv=tags.get('GPS GPSLongitude'); lonr=tags.get('GPS GPSLongitudeRef')
    if latv and lonv and latr and lonr:
        d['lat']=_dms(getattr(latv,'values',[]),latr)
        d['lon']=_dms(getattr(lonv,'values',[]),lonr)
    if d['lat'] is None or d['lon'] is None or d['yaw_deg'] is None or d['alt_m'] in (None,0.0):
        x=_read_dji_xmp(path)
        for k in ['lat','lon','yaw_deg','alt_m']:
            if d.get(k) in (None,0.0) and x.get(k) not in (None,0.0): d[k]=x[k]
    with Image.open(path) as im: d['w'],d['h']=im.size
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
