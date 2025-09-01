import numpy as np, cv2 as cv, torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

class FeatureMatcher:
    def __init__(self, device, max_kpts=2000):
        self.device=device
        self.sp=SuperPoint({'descriptor_dim':256,'nms_radius':4,'keypoint_threshold':0.005,'max_keypoints':int(max_kpts)}).to(device).eval()
        self.sg=SuperGlue({'weights':'outdoor','sinkhorn_iterations':50,'match_threshold':0.2}).to(device).eval()
    @torch.no_grad()
    def match(self,g0,g1,orb_fallback=False):
        def t(x): return torch.from_numpy(x)[None,None].float().to(self.device)/255.0
        o0,o1=self.sp({'image':t(g0)}),self.sp({'image':t(g1)})
        k0,k1=o0['keypoints'][0],o1['keypoints'][0]
        if k0.shape[0]==0 or k1.shape[0]==0: return self._orb(g0,g1) if orb_fallback else (np.zeros((0,2)),np.zeros((0,2)))
        data={'image0':t(g0),'image1':t(g1),'keypoints0':k0[None],'keypoints1':k1[None],'descriptors0':o0['descriptors'][0][None],'descriptors1':o1['descriptors'][0][None],'scores0':o0['scores'][0][None],'scores1':o1['scores'][0][None]}
        out=self.sg(data); m=out['matches0'][0].detach().cpu().numpy(); v=m>-1
        if not np.any(v): return self._orb(g0,g1) if orb_fallback else (np.zeros((0,2)),np.zeros((0,2)))
        p0=k0.detach().cpu().numpy()[v]; p1=k1.detach().cpu().numpy()[m[v]]
        return p0,p1
    def _orb(self,g0,g1,n=1500):
        orb=cv.ORB_create(nfeatures=int(n))
        k0,d0=orb.detectAndCompute(g0,None); k1,d1=orb.detectAndCompute(g1,None)
        if d0 is None or d1 is None or len(k0)<6 or len(k1)<6: return np.zeros((0,2)),np.zeros((0,2))
        m=sorted(cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True).match(d0,d1),key=lambda x:x.distance)[:800]
        if len(m)<6: return np.zeros((0,2)),np.zeros((0,2))
        p0=np.float32([k0[i.queryIdx].pt for i in m]); p1=np.float32([k1[i.trainIdx].pt for i in m])
        return p0,p1
