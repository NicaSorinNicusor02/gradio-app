import json, numpy as np, cv2 as cv

class FaultProjector:
    def __init__(self,index_json):
        idx=json.load(open(index_json,'r'))
        self.size=tuple(idx["mosaic_size"])
        self.Hp=[np.array(H,np.float64) for H in idx["H_proc_to_pano"]]
        self.frames=idx["frames"]

    def _raw_to_proc(self,i,pts_xy):
        f=self.frames[i]
        pts=np.asarray(pts_xy,np.float64).reshape(-1,1,2)
        K=f["K"]; D=f["D"]; newK=f["newK"]
        if K is not None and newK is not None:
            K=np.array(K,np.float64); D=np.array(D,np.float64).reshape(-1,1); newK=np.array(newK,np.float64)
            pn=cv.undistortPoints(pts, K, D, P=None)
            pn=pn.reshape(-1,2)
            ones=np.ones((pn.shape[0],1),np.float64)
            pix=(newK @ np.hstack([pn,ones]).T).T[:,0:2]
        else:
            pix=pts.reshape(-1,2)
        M=np.array(f["M_yaw"],np.float64)
        ones=np.ones((pix.shape[0],1),np.float64)
        rot=(M @ np.hstack([pix,ones]).T).T[:,0:2]
        s=float(f["scale"])
        proc=rot*np.array([s,s],np.float64)
        return proc

    def _proc_to_pano(self,i,pts):
        H=self.Hp[i]
        pts=np.asarray(pts,np.float64).reshape(-1,1,2)
        out=cv.perspectiveTransform(pts,H).reshape(-1,2)
        return out

    def raw_bbox_to_pano_poly(self,i,xyxy,samples=8):
        x1,y1,x2,y2=map(float,xyxy)
        xs=np.linspace(x1,x2,samples); ys=np.linspace(y1,y2,samples)
        top=np.stack([xs, np.full_like(xs,y1)],1)
        right=np.stack([np.full_like(ys,x2), ys],1)
        bottom=np.stack([xs[::-1], np.full_like(xs,y2)],1)
        left=np.stack([np.full_like(ys,y1), ys[::-1]],1)
        ring=np.vstack([top[:-1], right[:-1], bottom[:-1], left[:-1]])
        a=self._raw_to_proc(i, ring)
        b=self._proc_to_pano(i, a)
        return b

    @staticmethod
    def poly_to_aabb(poly):
        p=np.asarray(poly,np.float64)
        mn=p.min(0); mx=p.max(0)
        return [float(mn[0]),float(mn[1]),float(mx[0]),float(mx[1])]
