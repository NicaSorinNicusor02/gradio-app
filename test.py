#!/usr/bin/env python3
# test_simple.py
import os, sys, glob
import cv2 as cv
import torch

# --- încearcă YOLOv5 local (gradio-app/yolov5), cu fallback pe remote ---
ROOT = os.path.dirname(os.path.abspath(__file__))
Y5_DIR = os.path.join(ROOT, "yolov5")

def load_model(weights, size=512, conf=0.15, iou=0.45, device=""):
    w = os.path.abspath(weights)
    try:
        model = torch.hub.load(Y5_DIR, 'custom', w, source='local', force_reload=False)
    except Exception:
        model = torch.hub.load('ultralytics/yolov5', 'custom', w, trust_repo=True)
    model.to(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    # praguri
    try:
        model.conf = float(conf)
        model.iou = float(iou)
        model.max_det = 300
        model.classes = None
    except Exception:
        pass
    names = getattr(model, "names", None)
    names = names if isinstance(names, (list, tuple)) else list(names.values())
    return model, names, int(size)

def iter_images(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(path, "**", e), recursive=True)
    return sorted(files)

def main():
    import argparse
    ap = argparse.ArgumentParser("YOLOv5 quick test")
    ap.add_argument("-w","--weights", required=True, help="model .pt")
    ap.add_argument("-i","--images",  required=True, help="imagine sau folder cu imagini")
    ap.add_argument("-s","--size",    type=int, default=512, help="input size (default 512)")
    ap.add_argument("--conf",         type=float, default=0.15, help="confidence threshold")
    ap.add_argument("--iou",          type=float, default=0.45, help="IoU threshold")
    ap.add_argument("--device",       default="", help="cuda/cpu (auto dacă e gol)")
    args = ap.parse_args()

    model, names, size = load_model(args.weights, args.size, args.conf, args.iou, args.device)
    imgs = iter_images(args.images)
    if not imgs:
        print(f"[!] Nu am găsit imagini în {args.images}")
        return

    cv.namedWindow("detections", cv.WINDOW_NORMAL)
    for p in imgs:
        im = cv.imread(p, cv.IMREAD_COLOR)
        if im is None:
            print(f"[warn] nu pot citi {p}")
            continue

        # YOLOv5 AutoShape face letterbox intern; îi dăm RGB și cerem size=512
        res = model(im[:, :, ::-1], size=size)
        if hasattr(res, "xyxy"):
            det = res.xyxy[0].detach().cpu().numpy()
        else:
            det = res.pandas().xyxy[0].values

        print(f"{os.path.basename(p)} -> {0 if det is None else len(det)} boxes")

        # desenează
        if det is not None and len(det):
            for x1,y1,x2,y2,conf,cls in det:
                x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                label = names[int(cls)] if names else str(int(cls))
                cv.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2, cv.LINE_AA)
                cv.putText(im, f"{label} {float(conf):.2f}", (x1, max(0,y1-6)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv.LINE_AA)

        cv.imshow("detections", im)
        k = cv.waitKey(0) & 0xFF
        if k in (27, ord('q')):  # Esc sau q
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
