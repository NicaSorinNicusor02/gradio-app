#!/usr/bin/env python3
# core/detector.py
import os, json, warnings
from typing import List, Dict, Optional
import numpy as np, cv2 as cv, torch, yaml

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def _load_names_from_yaml(path: Optional[str]):
    if not path: return None
    y = yaml.safe_load(open(path, "r"))
    n = y.get("names", None)
    if n is None: return None
    if isinstance(n, dict): m=[v for k,v in sorted(n.items(), key=lambda kv:int(k))]
    elif isinstance(n,(list,tuple)): m=list(n)
    else: return None
    return [str(x) for x in m]

def _parse_classes(spec, names):
    if not spec or not names: return None
    ids, byname = [], {str(v):k for k,v in enumerate(names)}
    toks = spec.split(",") if isinstance(spec,str) else spec
    for t in toks:
        t = str(t).strip()
        if not t: continue
        if t.isdigit(): ids.append(int(t))
        elif t in byname: ids.append(byname[t])
    ids = sorted(set(i for i in ids if 0 <= i < len(names)))
    return ids if ids else None

class YOLODetector:
    """
    Wrapper that accepts ANY .pth trained on Ultralytics YOLO or YOLOv5.
    """
    def __init__(self, weights:str, device:str="", imgsz:int=640, conf:float=0.25, iou:float=0.45,
                 max_det:int=300, classes=None, data_yaml:Optional[str]=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz, self.conf, self.iou, self.max_det = int(imgsz), float(conf), float(iou), int(max_det)
        self.backend, self.model, self.names = None, None, None

        # Try Ultralytics first
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            nm = self.model.names
            self.names = nm if isinstance(nm,(list,tuple)) else list(nm.values())
            self.backend = "ultralytics"
        except Exception:
            # Fallback to yolov5 hub
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', weights, trust_repo=True)
                self.model.to(self.device)
                self.names = getattr(self.model, 'names', None) or getattr(self.model.module, 'names', None)
                self.backend = "yolov5_hub"
            except Exception as e:
                raise SystemExit(f"[detector] cannot load weights '{weights}': {e}")

        # Optionally override names with a data.yaml
        ynames = _load_names_from_yaml(data_yaml)
        if ynames and len(ynames)>=1: self.names = ynames
        self.class_ids = _parse_classes(classes, self.names)

    def infer_image(self, img_bgr: np.ndarray) -> List[Dict]:
        if img_bgr is None or img_bgr.size == 0: return []
        if self.backend == "ultralytics":
            res = self.model.predict(source=img_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     device=self.device, max_det=self.max_det, classes=self.class_ids, verbose=False)
            r = res[0]
            if r.boxes is None or r.boxes.data is None or len(r.boxes) == 0: return []
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
        else:
            self.model.conf=self.conf; self.model.iou=self.iou
            self.model.max_det=self.max_det; self.model.classes=self.class_ids
            pred = self.model(img_bgr, size=self.imgsz)
            p0 = pred.xyxy[0].cpu().numpy() if hasattr(pred,"xyxy") else pred.pandas().xyxy[0].values
            if p0 is None or len(p0)==0: return []
            xyxy = p0[:, :4]; conf = p0[:, 4]; cls = p0[:, 5].astype(int)

        out=[]
        for (x1,y1,x2,y2),sc,c in zip(xyxy,conf,cls):
            out.append({
                "bbox":[int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))],
                "score":float(sc),
                "cls":int(c),
                "label": str(self.names[int(c)]) if self.names else str(int(c))
            })
        return out

    def run_on_index(self, index_json:dict, limit:int=None, save_vis_dir:str=None) -> List[Dict]:
        frames = index_json["frames"][:limit] if limit else index_json["frames"]
        if save_vis_dir: os.makedirs(save_vis_dir, exist_ok=True)
        results=[]
        for i, f in enumerate(frames):
            im = cv.imread(f["path"], cv.IMREAD_COLOR)
            if im is None: continue
            dets = self.infer_image(im)
            for d in dets:
                results.append({"image_index":i, "bbox":d["bbox"], "label":d["label"],
                                "cls":d.get("cls",0), "score":d["score"]})
            if save_vis_dir:
                vis=im.copy()
                for d in dets:
                    x1,y1,x2,y2=d["bbox"]
                    cv.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2,cv.LINE_AA)
                    cv.putText(vis,f"{d['label']} {d['score']:.2f}",
                               (x1,y1-6),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv.LINE_AA)
                cv.imwrite(os.path.join(save_vis_dir, f"det_{i:05d}.jpg"), vis)
        return results
