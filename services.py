import os, json,  traceback
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2 as cv
import gradio as gr
import webbrowser 

from utils import (
    sorted_images, overlay_path, boxes_path, world_path, index_path, gcps_path,
    gcps_count
)

# ---- Detection (real or simulated) ----
from core.detector import YOLODetector
import math


def run_detector_or_sim(run: str, simulate: bool, weights: str, data_yaml: str,
                        classes: str, imgsz: int, conf: float, iou: float, max_det: int) -> Tuple[str, str]:
    if not run: return "Upload images first.", ""
    imgs_dir = os.path.join(run, "images")
    files = sorted_images(imgs_dir)
    if not files: return "No valid images.", ""
    if simulate: return "Simulation enabled. Use the annotator below, then Save.", ""
    if not weights: return "Provide YOLO weights or enable Simulation.", ""
    try:
        det = YOLODetector(weights=weights, imgsz=int(imgsz), conf=float(conf), iou=float(iou),
                           max_det=int(max_det), classes=classes, data_yaml=data_yaml)
        dets = []
        os.makedirs(os.path.join(run, "det_vis"), exist_ok=True)
        for i, fp in enumerate(files):
            im = cv.imread(fp, cv.IMREAD_COLOR)
            if im is None: continue
            out = det.infer_image(im)
            vis = im.copy()
            for d in out:
                x1, y1, x2, y2 = d["bbox"]
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3, cv.LINE_AA)
                cv.putText(vis, f"{d['label']} {d['score']:.2f}", (x1, y1 - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
                dets.append({"image_index": i, "bbox": d["bbox"], "label": d["label"], "score": float(d["score"])})
            cv.imwrite(os.path.join(run, "det_vis", f"det_{i:05d}.jpg"), vis)
        det_json = os.path.join(run, "detections.json")
        with open(det_json, "w") as f: json.dump(dets, f, indent=2)
        return f"Detector finished. {len(dets)} boxes across {len(files)} images.", det_json
    except Exception as e:
        return f"Detection failed: {e}", ""

# ---- Manual simulation of detections ----
def _draw_boxes_on_image(img_bgr, boxes: List[List[int]]):
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3, cv.LINE_AA)
    return out

def _load_run_frames(run: str):
    paths = sorted_images(os.path.join(run, "images"))
    names = [os.path.basename(p) for p in paths]
    return paths, names

def sim_init(run: str):
    if not run: return gr.Dropdown(choices=[], value=None), None, {}, "Upload images first."
    paths, names = _load_run_frames(run)
    if not paths: return gr.Dropdown(choices=[], value=None), None, {}, "No frames."
    im = cv.imread(paths[0], cv.IMREAD_COLOR)
    return gr.Dropdown(choices=names, value=names[0]), cv.cvtColor(im, cv.COLOR_BGR2RGB), {}, "Two-click: corner → corner."

def sim_change_frame(run: str, frame_name: str, sim_state: dict):
    paths, names = _load_run_frames(run)
    if frame_name not in names: return None, []
    fi = names.index(frame_name)
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    boxes = sim_state.get(fi, [])
    return cv.cvtColor(_draw_boxes_on_image(im, boxes), cv.COLOR_BGR2RGB), boxes

def sim_click(evt: gr.SelectData, run: str, frame_name: str, click_state: dict, sim_state: dict):
    paths, names = _load_run_frames(run)
    if frame_name not in names: return None, click_state, sim_state
    fi = names.index(frame_name)
    x, y = map(int, evt.index)
    click_state = click_state or {"active": False, "x0": 0, "y0": 0}
    sim_state = sim_state or {}
    if not click_state["active"]:
        click_state = {"active": True, "x0": x, "y0": y}
    else:
        x1, y1 = click_state["x0"], click_state["y0"]; x2, y2 = x, y
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        sim_state.setdefault(fi, []).append([x1, y1, x2, y2])
        click_state = {"active": False, "x0": 0, "y0": 0}
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    vis = _draw_boxes_on_image(im, sim_state.get(fi, []))
    return cv.cvtColor(vis, cv.COLOR_BGR2RGB), click_state, sim_state

def sim_undo(run: str, frame_name: str, sim_state: dict):
    paths, names = _load_run_frames(run)
    if frame_name not in names: return None, sim_state
    fi = names.index(frame_name)
    if sim_state.get(fi): sim_state[fi].pop()
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    vis = _draw_boxes_on_image(im, sim_state.get(fi, []))
    return cv.cvtColor(vis, cv.COLOR_BGR2RGB), sim_state

def sim_clear(run: str, frame_name: str, sim_state: dict):
    paths, names = _load_run_frames(run)
    if frame_name not in names: return None, sim_state
    fi = names.index(frame_name); sim_state[fi] = []
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB), sim_state

def sim_save(run: str, sim_state: dict, label: str, score: float):
    if not run: return "Upload images first.", ""
    dets = []
    for i, boxes in (sim_state or {}).items():
        for b in boxes:
            dets.append({
                "image_index": int(i),
                "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                "label": label or "manual",
                "score": float(score),
            })
    det_json = os.path.join(run, "detections.json")
    with open(det_json, "w") as f: json.dump(dets, f, indent=2)
    return f"Saved {len(dets)} boxes to detections.json", det_json

# ---- Stitch + geo + projection ----
from core.stitcher import Stitcher
from core.mapper import GeoMapper
from core.box_projector import PanoBoxProjector

def do_stitch_and_project(run: str, reuse: bool=True):
    """
    Stitch + make world files even with 0 detections.
    Always return a valid image filepath (or None) for the Gradio Image.
    """
    if not run:
        return None, "Upload images first.", "", ""

    imgs_dir = os.path.join(run, "images")
    if not os.path.isdir(imgs_dir) or not os.listdir(imgs_dir):
        return None, "No images.", "", ""

    pano = os.path.join(run, "pano.png")
    det_json = os.path.join(run, "detections.json")
    idx_p = index_path(pano)
    world_p = world_path(pano)
    ov_p = overlay_path(pano)

    try:
        need_build = True
        if reuse and os.path.isfile(pano) and os.path.isfile(idx_p) and os.path.isfile(world_p):
            need_build = False

        if need_build:
            stitcher = Stitcher(
                imgs_dir=imgs_dir, max_kpts=2000, yaw_align=True,
                exposure=False, no_seams=False, num_bands=3, orb_fallback=False
            )
            pano_img, meta, exifs = stitcher.run()

            gmap = GeoMapper(align='yaw')
            phi_deg, A_world, epsg = gmap.choose_rotation(meta, exifs)
            pano_oriented, A_adj, epsg_out, world_json_path, M_pano, _ = gmap.orient_and_world(
                pano_img, phi_deg, A_world, epsg, pano
            )
            if not cv.imwrite(pano, pano_oriented):
                raise RuntimeError("Failed to save panorama")

            # Build index.json for projector
            Hs = [np.array(H, np.float64) for H in meta["Hs_shift"]]
            M_pano = np.array(M_pano, np.float64) if M_pano is not None else np.eye(3)
            Hs_out = [(M_pano @ H).tolist() for H in Hs]
            index = {
                "mosaic_size": [int(pano_oriented.shape[1]), int(pano_oriented.shape[0])],
                "H_proc_to_pano": Hs_out,
                "frames": meta.get("frame_models", []),
            }
            with open(idx_p, "w") as f:
                json.dump(index, f, indent=2)

        # Projection (works with empty/missing detections.json)
        idx = json.load(open(idx_p, "r"))
        dets = json.load(open(det_json, "r")) if os.path.isfile(det_json) else []

        pano_bgr = cv.imread(pano, cv.IMREAD_COLOR)
        if pano_bgr is None:
            raise RuntimeError("Cannot read panorama after stitching.")

        pj = PanoBoxProjector(idx, pano_bgr=pano_bgr)
        boxes_on_pano = pj.project(dets)

        # Write overlay; if it fails, copy pano to overlay
        try:
            pj.write_overlay(ov_p)
            if not os.path.isfile(ov_p):
                raise RuntimeError("write_overlay didn't produce a file")
        except Exception:
            cv.imwrite(ov_p, pano_bgr)  # fallback

        with open(boxes_path(pano), "w") as f:
            json.dump(boxes_on_pano, f, indent=2)

        # ---- Normalize returns
        ov_abs   = os.path.abspath(ov_p) if ov_p else ""
        pano_abs = os.path.abspath(pano) if pano else ""
        img_out  = ov_abs if (ov_abs and os.path.isfile(ov_abs)) else None  # <- crucial

        msg = f"Projected {len(boxes_on_pano)} boxes." if dets else \
              "No detections found — stitched panorama and world files are ready."

        return img_out, msg, ov_abs, pano_abs

    except Exception as e:
        pano_abs = os.path.abspath(pano) if os.path.isfile(pano) else ""
        # Never return "" for the Image component — return None instead
        return None, f"Processing failed: {e}", "", pano_abs

# ---- Geolocalization (click-to-lat/lon) ----
from core.geolocalizer import PanoGeoLocalizer

def map_click(evt: gr.SelectData, run: str, open_maps: bool, zoom: int, use_gcps: bool):
    if not run:
        return "Upload images first."
    pano = os.path.join(run, "pano.png")
    gcps = gcps_path(run) if (use_gcps and os.path.exists(gcps_path(run))) else None
    try:
        gl = PanoGeoLocalizer(
            pano,
            world_json=world_path(pano),
            gcps_json=gcps,
            boxes_json=boxes_path(pano),
            open_on_click=False,   # we handle opening here
            zoom=zoom
        )
    except Exception as e:
        return f"{e}"

    # Click position (u, v) -> lat/lon
    u, v = map(float, evt.index)
    lat, lon, E, N = gl.uv_to_latlon(u, v)

    # Compose the Google Maps URL
    url = f"https://www.google.com/maps/search/?api=1&query={lat:.7f}%2C{lon:.7f}&zoom={int(zoom)}"

    # Open a new browser tab if the toggle is ON (restores previous behavior)
    if open_maps:
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            # Ignore environments where opening a local browser isn't possible
            pass

    tail = f" · GCPs used: {gcps_count(gcps)}" if gcps else " · GCPs: off"
    return f"lat/lon: `{lat:.7f}, {lon:.7f}`{tail}  ·  [Open in Google Maps]({url})"
# ---- Summaries / reports / faults browser ----
def _severity(score: float) -> str:
    return "High" if score >= 0.80 else ("Medium" if score >= 0.50 else "Low")

def friendly_summary(run: str):
    if not run: return "Upload images first.", []
    p = os.path.join(run, "detections.json")
    if not os.path.exists(p): return "No detections yet.", []
    dets = json.load(open(p, "r"))
    by_cls: Dict[str, int] = {}
    for d in dets:
        by_cls[d.get("label", "fault")] = by_cls.get(d.get("label", "fault"), 0) + 1
    header = [f"Detected {len(dets)} faults"]
    if by_cls: header.append("Classes: " + ", ".join([f"{k}: {v}" for k, v in sorted(by_cls.items())]))
    rows = [[d.get("image_index", -1), d.get("label","fault"),
             _severity(float(d.get("score",1.0))), "Analysis placeholder"] for d in dets[:300]]
    return "\n".join(header), rows

from PIL import Image, ImageDraw

def pdf_report(run: str) -> Optional[str]:
    if not run: return None
    p = os.path.join(run, "detections.json")
    if not os.path.exists(p): return None
    detections = json.load(open(p, "r"))
    files = sorted_images(os.path.join(run, "images"))
    pages = []
    for i, fp in enumerate(files):
        im = Image.open(fp).convert("RGB")
        draw = ImageDraw.Draw(im)
        boxes = [d for d in detections if d["image_index"] == i]
        for d in boxes:
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            draw.text((x1 + 8, max(5, y1 - 24)),
                      f"{d.get('label','fault')} {float(d.get('score',1.0)):.2f}",
                      fill=(0, 255, 0))
        draw.text((10, 10), f"Frame {i}", fill=(255, 0, 0))
        pages.append(im)
    if not pages: return None
    out_pdf = os.path.join(run, "report.pdf")
    pages[0].save(out_pdf, "PDF", resolution=100.0, save_all=True, append_images=pages[1:])
    return out_pdf

def faults_table(run: str):
    if not run: return [], "Upload images first.", ""
    p = os.path.join(run, "detections.json")
    if not os.path.exists(p): return [], "No detections.", ""
    dets = json.load(open(p, "r"))
    files = sorted_images(os.path.join(run, "images"))
    rows = []
    for d in dets:
        idx = d["image_index"]
        img_name = os.path.basename(files[idx]) if 0 <= idx < len(files) else f"IMG{idx}"
        rows.append([img_name, d.get("label","fault"),
                     f"{float(d.get('score',1.0)):.2f}",
                     _severity(float(d.get('score',1.0))),
                     "Analysis placeholder"])
    msg = f"{len(rows)} detections."
    first_img = files[dets[0]["image_index"]] if dets else ""
    return rows, msg, first_img

def build_fault_index(run: str):
    p = os.path.join(run, "detections.json")
    if not os.path.exists(p): return [], {}
    dets = json.load(open(p, "r"))
    files = sorted_images(os.path.join(run, "images"))
    index: Dict[int, List[dict]] = {}
    for d in dets:
        idx = d["image_index"]
        index.setdefault(idx, []).append(d)
    names = []
    for idx in sorted(index.keys()):
        name = os.path.basename(files[idx]) if 0 <= idx < len(files) else f"IMG{idx}"
        names.append((name, idx))
    return names, index, files

def render_fault_image(files: List[str], idx: int, det_index: Dict[int, List[dict]]):
    if not files or idx is None or idx < 0 or idx >= len(files): return None, []
    im = cv.imread(files[idx], cv.IMREAD_COLOR)
    rows = []
    for d in det_index.get(idx, []):
        x1, y1, x2, y2 = d["bbox"]
        cv.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3, cv.LINE_AA)
        cv.putText(im, f"{d.get('label','fault')} {float(d.get('score',1.0)):.2f}",
                   (x1, y1 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        rows.append([os.path.basename(files[idx]), d.get("label","fault"),
                     f"{float(d.get('score',1.0)):.2f}",
                     _severity(float(d.get('score',1.0))),
                     "Analysis placeholder"])
    return cv.cvtColor(im, cv.COLOR_BGR2RGB), rows
