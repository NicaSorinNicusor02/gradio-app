import os, glob, numpy as np, cv2 as cv, torch
from tqdm import tqdm
from .exif_utils import read_meta, sensor_width_mm, gsd_cm_per_px, intrinsics
from .feature_matcher import FeatureMatcher
from .geometry import yaw_level, rescale_to_gsd, robust_H
from .warper import blend_warped

class Stitcher:
    def __init__(self, imgs_dir, max_kpts=2000, yaw_align=False, exposure=False, no_seams=False, num_bands=3, orb_fallback=False):
        self.imgs_dir=imgs_dir; self.max_kpts=max_kpts; self.yaw_align=yaw_align; self.exposure=exposure
        self.use_seams=not no_seams; self.num_bands=num_bands; self.orb_fallback=orb_fallback
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.matcher=FeatureMatcher(self.device,max_kpts=max_kpts)
    def _paths(self):
        p=sorted(glob.glob(os.path.join(self.imgs_dir,'*.[jJ][pP][gG]'))+glob.glob(os.path.join(self.imgs_dir,'*.[Jj][Pp][Ee][Gg]'))+glob.glob(os.path.join(self.imgs_dir,'*.[pP][nN][gG]')))
        if len(p)<2: raise SystemExit('need at least two images'); return p
        return p
    def run(self):
        paths=self._paths(); exifs=[read_meta(p) for p in paths]
        sw0=sensor_width_mm(exifs[0]['focal_mm'],exifs[0]['focal35mm'])
        gsd=[gsd_cm_per_px(e['alt_m'],e['focal_mm'],sw0,e['w']) for e in exifs]
        tg=float(np.median([g for g in gsd if g is not None])) if any(g is not None for g in gsd) else None
        yaws=[e['yaw_deg'] for e in exifs if e['yaw_deg'] is not None]; yref=float(np.median(yaws)) if (yaws and self.yaw_align) else None
        imgs=[]; frame_models=[]
        for path,e in zip(paths,exifs):
            im=cv.imread(path,cv.IMREAD_COLOR); 
            if im is None: continue
            raw_h,raw_w=im.shape[:2]
            K,D=intrinsics(e)
            if K is not None:
                newK,_=cv.getOptimalNewCameraMatrix(K,D,(raw_w,raw_h),alpha=0.0,newImgSize=(raw_w,raw_h))
                im=cv.undistort(im,K,D,None,newK)
            else: newK=None
            yaw_delta=float((e['yaw_deg']-yref)) if (self.yaw_align and e['yaw_deg'] is not None) else 0.0
            if self.yaw_align:
                c=(raw_w//2,raw_h//2); M_yaw=cv.getRotationMatrix2D(c,yaw_delta,1.0)
                im=cv.warpAffine(im,M_yaw,(raw_w,raw_h),flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT)
            else: M_yaw = np.eye(2, 3, dtype=np.float64)
            g=gsd_cm_per_px(e['alt_m'],e['focal_mm'],sw0,e['w']); im,s=rescale_to_gsd(im,g,tg)
            proc_h,proc_w=im.shape[:2]; imgs.append(im)
            frame_models.append({"path":path,"raw_size":[int(raw_w),int(raw_h)],"proc_size":[int(proc_w),int(proc_h)],"K":(K.tolist() if K is not None else None),"D":(D.reshape(-1).tolist() if K is not None else None),"newK":(newK.tolist() if newK is not None else None),"yaw_delta_deg":float(yaw_delta),"M_yaw":M_yaw.tolist(),"scale":float(s)})
        if len(imgs)<2: raise SystemExit('not enough valid images')
        def gray_small(bgr): g=cv.cvtColor(bgr,cv.COLOR_BGR2GRAY); return cv.resize(g,None,fx=0.5,fy=0.5,interpolation=cv.INTER_AREA),0.5
        gs=[gray_small(im) for im in imgs]
        Hs=[]
        for i in tqdm(range(len(imgs)-1),desc='Matching & H'):
            g0,s0=gs[i]; g1,s1=gs[i+1]; p0,p1=self.matcher.match(g0,g1,orb_fallback=self.orb_fallback)
            if len(p0)<6: raise RuntimeError(f'not enough matches for pair {i}-{i+1}')
            H=robust_H(p0/s0,p1/s1); 
            if H is None or not np.all(np.isfinite(H)): raise RuntimeError(f'homography failed for {i}-{i+1}')
            Hs.append(H)
        H_0k=[np.eye(3,dtype=np.float64)]
        for H in Hs: H_0k.append(H_0k[-1]@H)
        H_ref=[np.linalg.inv(H) for H in H_0k]
        pano,meta=blend_warped(imgs,H_ref,use_exposure=self.exposure,use_seams=self.use_seams,num_bands=self.num_bands)
        meta["frame_models"]=frame_models
        return pano,meta,exifs
