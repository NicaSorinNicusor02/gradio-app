#!/usr/bin/env python3
# core/box_projector.py
import os, json
from typing import List, Dict
import numpy as np, cv2 as cv

def _raw_to_proc(frame:dict, pts:np.ndarray) -> np.ndarray:
    K = np.array(frame["K"], dtype=np.float64)
    D = np.array(frame["D"], dtype=np.float64)
    newK = np.array(frame["newK"], dtype=np.float64)
    pts = cv.undistortPoints(np.expand_dims(pts,1), K, D, P=newK).reshape(-1,2)
    M_yaw = np.array(frame["M_yaw"], dtype=np.float64)
    pts = cv.transform(np.expand_dims(pts,1), M_yaw).reshape(-1,2)
    s = float(frame["scale"])
    pts *= s
    return pts

def _project_box_to_pano(frame:dict, H_proc_to_pano:np.ndarray, box:list) -> np.ndarray:
    x1,y1,x2,y2 = map(float, box)
    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float64)
    proc = _raw_to_proc(frame, corners)
    H = np.array(H_proc_to_pano, dtype=np.float64)
    H /= H[2,2]
    pano_pts = cv.perspectiveTransform(proc[None,:,:], H)[0]
    return pano_pts  # (4,2)

class PanoBoxProjector:
    """
    Projects detections from raw frame pixel coords onto the panorama.
    """
    def __init__(self, index_json:dict, pano_bgr=None):
        self.idx = index_json
        self.frames = index_json["frames"]
        self.Hs = [np.array(H, dtype=np.float64) for H in index_json["H_proc_to_pano"]]
        self.pano = pano_bgr  # optional (for drawing)

    def project(self, detections:List[Dict]) -> List[Dict]:
        out=[]
        for d in detections:
            i = int(d["image_index"])
            if i<0 or i>=len(self.frames): continue
            frame = self.frames[i]
            H = self.Hs[i]
            quad = _project_box_to_pano(frame, H, d["bbox"])
            out.append({
                "id": d.get("id", f"{i}"),
                "image_index": i,
                "quad": quad.astype(int).tolist(),
                "label": d.get("label",""),
                "score": float(d.get("score",1.0))
            })
            if self.pano is not None:
                cv.polylines(self.pano,[quad.astype(int)],True,(0,255,0),3,cv.LINE_AA)
                cx,cy = int(quad[:,0].mean()), int(quad[:,1].mean())
                cv.putText(self.pano, f'#{d.get("id",i)}', (cx,cy),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv.LINE_AA)
        return out

    def write_overlay(self, path:str):
        if self.pano is not None:
            cv.imwrite(path, self.pano)

    @staticmethod
    def project_and_save(index_path:str, detections_path:str, pano_path:str,
                         out_overlay:str="pano_overlay.png", out_boxes:str="pano.boxes.json"):
        idx = json.load(open(index_path, "r"))
        pano = cv.imread(pano_path, cv.IMREAD_COLOR)
        if pano is None: raise SystemExit(f"cannot read pano '{pano_path}'")
        dets = json.load(open(detections_path, "r"))
        pj = PanoBoxProjector(idx, pano_bgr=pano)
        boxes = pj.project(dets)
        pj.write_overlay(out_overlay)
        with open(out_boxes, "w") as f: json.dump(boxes, f, indent=2)
        print(f"[project] wrote {out_overlay}, {out_boxes}  (n={len(boxes)})")
