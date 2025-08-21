# scripts/stitch_superglue.py
import os, sys, glob
from typing import List, Tuple
import numpy as np
import cv2
import torch

SUPERGLUE_ROOT = r"C:\Users\Sorin\Desktop\poze_enevo\proiect\SuperGluePretrainedNetwork"
if SUPERGLUE_ROOT not in sys.path:
    sys.path.insert(0, SUPERGLUE_ROOT)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue

def build_panorama(
    image_dir: str,
    superpoint_weights: str,
    superglue_weights: str,
    project_root: str,
    output_path: str,
    target_h: int = 1200,
    target_w: int = 1600,
) -> str:
    def device_select() -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nu este disponibilă. Instalează PyTorch cu CUDA + driver NVIDIA compatibil.")
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")

    def load_image_gray(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            raise FileNotFoundError(f"Nu pot citi imaginea: {path}")
        img_color = cv2.resize(img_color, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return img_color, img_gray

    def npimg_to_torch(img_gray: np.ndarray, device: torch.device) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(img_gray)).unsqueeze(0).unsqueeze(0)
        return t.to(device)

    def build_models(sp_weights: str, sg_weights: str, device: torch.device):
        sp_config = {'descriptor_dim':256, 'nms_radius':4, 'keypoint_threshold':0.005, 'max_keypoints':4096}
        sg_config = {'descriptor_dim':256, 'weights':'outdoor', 'sinkhorn_iterations':20, 'match_threshold':0.2}
        sp = SuperPoint(sp_config).to(device).eval()
        sg = SuperGlue(sg_config).to(device).eval()
        sp_sd = torch.load(sp_weights, map_location=device)
        sg_sd = torch.load(sg_weights, map_location=device)
        sp.load_state_dict(sp_sd.get('model', sp_sd), strict=False)
        sg.load_state_dict(sg_sd.get('model', sg_sd), strict=False)
        return sp, sg

    def extract_features(sp, img_gray, device):
        with torch.no_grad():
            tens = npimg_to_torch(img_gray, device)
            out = sp({'image': tens})
            kpts  = out['keypoints'][0].detach().cpu().numpy()
            scores= out['scores'][0].detach().cpu().numpy()
            desc  = out['descriptors'][0].detach().cpu().numpy()
        return kpts, scores, desc

    def superglue_match(sg, k0,s0,d0, k1,s1,d1, t0,t1, device):
        data = {
            'keypoints0':   torch.from_numpy(k0)[None].float().to(device),
            'keypoints1':   torch.from_numpy(k1)[None].float().to(device),
            'scores0':      torch.from_numpy(s0)[None].float().to(device),
            'scores1':      torch.from_numpy(s1)[None].float().to(device),
            'descriptors0': torch.from_numpy(d0)[None].float().to(device),
            'descriptors1': torch.from_numpy(d1)[None].float().to(device),
            'image0': t0, 'image1': t1,
        }
        with torch.no_grad():
            pred = sg(data)
            matches0 = pred['matches0'][0].detach().cpu().numpy()
        valid = matches0 > -1
        return k0[valid], k1[matches0[valid]]

    def estimate_homography(mk0: np.ndarray, mk1: np.ndarray) -> np.ndarray:
        if len(mk0) < 4:
            return None
        H, _ = cv2.findHomography(mk1, mk0, cv2.RANSAC, 3.0)
        return H

    def compute_output_bounds(images: List[np.ndarray], Hs: List[np.ndarray]):
        allc = []
        for img, H in zip(images, Hs):
            h, w = img.shape[:2]
            c = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
            c = cv2.perspectiveTransform(c[None], H)[0]
            allc.append(c)
        allc = np.vstack(allc)
        mn = np.floor(allc.min(0)).astype(int)
        mx = np.ceil(allc.max(0)).astype(int)
        offset = (-mn[0], -mn[1])
        W = int(mx[0]-mn[0]); Hh = int(mx[1]-mn[1])
        return (W, Hh), offset

    def warp_blend_hard(canvas, weight, img, H, offset, out_size):
        T = np.array([[1,0,offset[0]],[0,1,offset[1]],[0,0,1]], dtype=np.float32) @ H
        warped = cv2.warpPerspective(img, T, out_size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        mask8 = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8)*255, T, out_size)
        dt = cv2.distanceTransform((mask8>0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        if dt.max() > 0: dt /= dt.max()
        sel = dt > weight
        if np.any(sel):
            canvas[sel] = warped[sel]
            weight[sel] = dt[sel]

    def auto_crop_black(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        if not np.any(mask): return img
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        return img[y0:y1, x0:x1]

    # ---- pipeline ----
    device = device_select()

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.JPG","*.PNG","*.JPEG")
    paths = []
    for e in exts: paths += glob.glob(os.path.join(image_dir, e))
    paths = sorted({p.lower():p for p in paths}.values())  # case-insensitive unique
    if len(paths) < 2:
        raise RuntimeError("Pune cel puțin 2 imagini în folder.")

    sp, sg = build_models(superpoint_weights, superglue_weights, device)

    imgs_c, feats = [], []
    for p in paths:
        c, g = load_image_gray(p)
        k, s, d = extract_features(sp, g, device)
        t = npimg_to_torch(g, device)
        imgs_c.append(c); feats.append((k,s,d,t))

    pairH = [None]*len(paths)
    for i in range(1, len(paths)):
        k0,s0,d0,t0 = feats[i-1]
        k1,s1,d1,t1 = feats[i]
        mk0, mk1 = superglue_match(sg, k0,s0,d0, k1,s1,d1, t0,t1, device)
        H = estimate_homography(mk0, mk1)
        if H is None:
            raise RuntimeError(f"Nu s-a putut estima omografia între {i-1} și {i}.")
        pairH[i] = H

    ref = len(paths)//2
    H_to_ref = [None]*len(paths)
    H_to_ref[ref] = np.eye(3, dtype=np.float64)
    for i in range(ref+1, len(paths)): H_to_ref[i] = H_to_ref[i-1] @ pairH[i]
    for i in range(ref-1, -1, -1):     H_to_ref[i] = H_to_ref[i+1] @ np.linalg.inv(pairH[i+1])

    out_size, offset = compute_output_bounds(imgs_c, H_to_ref)
    W, Hh = out_size
    canvas = np.zeros((Hh, W, 3), np.float32)
    weight = np.zeros((Hh, W), np.float32)

    for img, H in zip(imgs_c, H_to_ref):
        warp_blend_hard(canvas, weight, img, H, offset, (W, Hh))

    pano = np.clip(canvas, 0, 255).astype(np.uint8)
    pano = auto_crop_black(pano)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    cv2.imwrite(output_path, pano)
    return output_path
