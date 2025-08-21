import numpy as np, cv2 as cv

def blend_warped(images,Hs,use_exposure=False,use_seams=True,num_bands=3,max_tile_side=8000):
    polys=[]
    for img,H in zip(images,Hs):
        h,w=img.shape[:2]
        c=np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]],np.float64).T
        p=H@c; p=(p[:2]/p[2]).T; p=p[np.isfinite(p).all(1)]
        polys.append(p if len(p) else None)
    valid=[p for p in polys if p is not None]
    if not valid: raise RuntimeError("no projected corners")
    allp=np.vstack(valid)
    minx,miny=np.floor(allp.min(0)).astype(int); maxx,maxy=np.ceil(allp.max(0)).astype(int)
    T=np.array([[1,0,-minx],[0,1,-miny],[0,0,1]],np.float64)
    Hs_shift=[T@H for H in Hs]
    corners=[]; tiles=[]; masks=[]; rois=[]
    for img,Hi in zip(images,Hs_shift):
        h,w=img.shape[:2]
        c=np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]],np.float64).T
        p=Hi@c; p=(p[:2]/p[2]).T; p=p[np.isfinite(p).all(1)]
        if p is None or len(p)==0: continue
        x,y,w,h=cv.boundingRect(p.astype(np.float32))
        if w<=0 or h<=0 or w>max_tile_side or h>max_tile_side: continue
        Ti=np.array([[1,0,-x],[0,1,-y],[0,0,1]],np.float64); Hroi=Ti@Hi
        warp=cv.warpPerspective(img,Hroi,(w,h),flags=cv.INTER_LINEAR,borderMode=cv.BORDER_TRANSPARENT)
        if warp.ndim==2: warp=cv.cvtColor(warp,cv.COLOR_GRAY2BGR)
        m=((warp[...,0]>0)|(warp[...,1]>0)|(warp[...,2]>0)).astype(np.uint8)*255
        if cv.countNonZero(m)==0: continue
        corners.append((int(x),int(y))); tiles.append(warp.astype(np.uint8,copy=False)); masks.append(m.astype(np.uint8,copy=False))
        rois.append((int(x),int(y),int(w),int(h)))
    if not tiles: raise RuntimeError("empty warps")
    comp=None
    if use_exposure:
        try: comp=cv.detail_GainCompensator(); comp.feed(corners,tiles,masks)
        except: comp=None
    if use_seams:
        try:
            seam=cv.detail_DpSeamFinder('COLOR'); seam_masks=seam.find(tiles,list(corners),masks)
            if seam_masks is None or len(seam_masks)!=len(masks): seam_masks=masks
        except: seam_masks=masks
    else: seam_masks=masks
    x0=min(x for x,y,w,h in rois); y0=min(y for x,y,w,h in rois)
    x1=max(x+w for x,y,w,h in rois); y1=max(y+h for x,y,w,h in rois)
    blender=cv.detail_MultiBandBlender(0,int(num_bands)); blender.prepare((int(x0),int(y0),int(x1-x0),int(y1-y0)))
    for i,(im,msk,cor) in enumerate(zip(tiles,seam_masks,corners)):
        if comp is not None:
            try: comp.apply(i,cor,im,msk)
            except: comp=None
        blender.feed(im.astype(np.int16),msk,cor)
    res,_=blender.blend(None,None)
    pano=np.clip(res,0,255).astype(np.uint8) if res.dtype!=np.uint8 else res
    gray=cv.cvtColor(pano,cv.COLOR_BGR2GRAY); nz=cv.findNonZero((gray>0).astype(np.uint8))
    if nz is None: return pano,{'Hs_shift':Hs_shift,'T_canvas':T,'crop':(0,0,pano.shape[1],pano.shape[0])}
    x,y,w,h=cv.boundingRect(nz); Tc=np.array([[1,0,-x],[0,1,-y],[0,0,1]],np.float64); H_out=[Tc@H for H in Hs_shift]
    return pano[y:y+h,x:x+w],{'Hs_shift':H_out,'T_canvas':T,'crop':(int(x),int(y),int(w),int(h))}
