#!/usr/bin/env python3
import os, sys, json, time, uuid, shutil, webbrowser, atexit, signal
from typing import List, Dict, Tuple, Optional

import gradio as gr
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from core.stitcher import Stitcher
from core.mapper import GeoMapper
from core.detector import YOLODetector
from core.box_projector import PanoBoxProjector
from core.geolocalizer import PanoGeoLocalizer

RUNS_DIR = os.path.join(HERE, "app_runs")
if os.path.exists(RUNS_DIR):
    shutil.rmtree(RUNS_DIR, ignore_errors=True)
os.makedirs(RUNS_DIR, exist_ok=True)

def _cleanup_runs(*_):
    shutil.rmtree(RUNS_DIR, ignore_errors=True)
signal.signal(getattr(signal, "SIGINT", 2), _cleanup_runs)
signal.signal(getattr(signal, "SIGTERM", 15), _cleanup_runs)
atexit.register(_cleanup_runs)

def _new_run_dir() -> str:
    d = os.path.join(RUNS_DIR, time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "det_vis"), exist_ok=True)
    return d

def _overlay_path(pano_png: str) -> str:
    return os.path.splitext(pano_png)[0] + ".overlay.png"

def _boxes_path(pano_png: str) -> str:
    return os.path.splitext(pano_png)[0] + ".boxes.json"

def _world_path(pano_png: str) -> str:
    return os.path.splitext(pano_png)[0] + ".world.json"

def _index_path(pano_png: str) -> str:
    return os.path.splitext(pano_png)[0] + ".index.json"

def _gcps_path(run_dir: str) -> str:
    return os.path.join(run_dir, "gcps.json")

def _gcps_count(path: Optional[str]) -> int:
    try:
        if not path: return 0
        g = json.load(open(path, "r"))
        return sum(1 for r in g if all(k in r for k in ("u","v","lat","lon")))
    except Exception:
        return 0

def _maps_url(lat: float, lon: float, zoom: int = 20) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.7f}%2C{lon:.7f}&zoom={zoom}"

def _severity(score: float) -> str:
    if score >= 0.80: return "High"
    if score >= 0.50: return "Medium"
    return "Low"

def _sorted_images(imgs_dir: str) -> List[str]:
    if not os.path.isdir(imgs_dir): return []
    allf = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir)
            if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"))]
    return sorted(allf, key=lambda p: os.path.basename(p).lower())

def dl_validate_and_copy(files: List[gr.File]) -> Tuple[str, str]:
    if not files:
        return "", "No files uploaded."
    run = _new_run_dir()
    imgs_dir = os.path.join(run, "images")
    ok, bad = 0, []
    for f in files:
        name = os.path.basename(getattr(f, "name", "img.jpg"))
        if not name.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp")):
            bad.append(f"{name}: unsupported extension"); continue
        dst = os.path.join(imgs_dir, name)
        try:
            shutil.copy(f.name, dst)
            im = cv.imread(dst, cv.IMREAD_COLOR)
            if im is None or im.size == 0:
                bad.append(f"{name}: corrupted/unreadable"); os.remove(dst)
            else:
                ok += 1
        except Exception as e:
            bad.append(f"{name}: copy/read error ({e})")
    msg = [f"Copied {ok} images.", f"Run: `{run}`"]
    if bad: msg.append("Issues:\n" + "\n".join([f"- {x}" for x in bad]))
    return run, "\n".join(msg)

def dl_attach_gcps(run: str, gcps_file: Optional[gr.File]) -> Tuple[str, str]:
    if not run: return "", "Upload images first."
    if gcps_file is None: return "", "No GCP file provided."
    try:
        dst = _gcps_path(run)
        shutil.copy(gcps_file.name, dst)
        return dst, f"GCPs saved ({_gcps_count(dst)} points)."
    except Exception as e:
        return "", f"Failed to save GCPs: {e}"

def dl_run_detector_or_sim(run: str, simulate: bool, weights: str,
                           data_yaml: str, classes: str, imgsz: int,
                           conf: float, iou: float, max_det: int) -> Tuple[str, str]:
    if not run: return "Upload images first.", ""
    imgs_dir = os.path.join(run, "images")
    files = _sorted_images(imgs_dir)
    if not files: return "No valid images.", ""
    if simulate: return "Simulation enabled. Use the annotator below, then Save.", ""
    if not weights: return "Provide YOLO weights or enable Simulation.", ""
    try:
        det = YOLODetector(weights=weights, imgsz=int(imgsz), conf=float(conf),
                           iou=float(iou), max_det=int(max_det), classes=classes, data_yaml=data_yaml)
        dets = []
        os.makedirs(os.path.join(run, "det_vis"), exist_ok=True)
        for i, fp in enumerate(files):
            im = cv.imread(fp, cv.IMREAD_COLOR)
            if im is None: continue
            out = det.infer_image(im)
            vis = im.copy()
            for d in out:
                x1,y1,x2,y2 = d["bbox"]
                cv.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3,cv.LINE_AA)
                cv.putText(vis,f"{d['label']} {d['score']:.2f}",(x1,y1-6),
                           cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv.LINE_AA)
                dets.append({"image_index": i, "bbox": d["bbox"],
                             "label": d["label"], "score": float(d["score"])})
            cv.imwrite(os.path.join(run, "det_vis", f"det_{i:05d}.jpg"), vis)
        det_json = os.path.join(run, "detections.json")
        with open(det_json, "w") as f: json.dump(dets, f, indent=2)
        return f"Detector finished. {len(dets)} boxes across {len(files)} images.", det_json
    except Exception as e:
        return f"Detection failed: {e}", ""

def _load_run_frames(run: str) -> Tuple[List[str], List[str]]:
    imgs_dir = os.path.join(run, "images")
    paths = _sorted_images(imgs_dir)
    names = [os.path.basename(p) for p in paths]
    return paths, names

def _draw_boxes_on_image(img_bgr: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
    out = img_bgr.copy()
    for (x1,y1,x2,y2) in boxes:
        cv.rectangle(out,(x1,y1),(x2,y2),(0,255,0),3,cv.LINE_AA)
    return out

def sim_init(run: str):
    if not run: return gr.Dropdown(choices=[], value=None), None, [], "Upload images first."
    paths, names = _load_run_frames(run)
    if not paths: return gr.Dropdown(choices=[], value=None), None, [], "No frames."
    im = cv.imread(paths[0], cv.IMREAD_COLOR)
    return gr.Dropdown(choices=names, value=names[0]), cv.cvtColor(im, cv.COLOR_BGR2RGB), [], "Two-click: corner → corner."

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
    if not click_state: click_state = {"active": False, "x0": 0, "y0": 0}
    if not sim_state: sim_state = {}
    if not click_state["active"]:
        click_state = {"active": True, "x0": x, "y0": y}
    else:
        x1,y1 = click_state["x0"], click_state["y0"]; x2,y2 = x, y
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        sim_state.setdefault(fi, []).append([x1,y1,x2,y2])
        click_state = {"active": False, "x0": 0, "y0": 0}
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    vis = _draw_boxes_on_image(im, sim_state.get(fi, []))
    return cv.cvtColor(vis, cv.COLOR_BGR2RGB), click_state, sim_state

def sim_undo(run: str, frame_name: str, sim_state: dict):
    paths, names = _load_run_frames(run)
    if frame_name not in names: return None, [], sim_state
    fi = names.index(frame_name)
    if sim_state.get(fi): sim_state[fi].pop()
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    vis = _draw_boxes_on_image(im, sim_state.get(fi, []))
    return cv.cvtColor(vis, cv.COLOR_BGR2RGB), sim_state.get(fi, []), sim_state

def sim_clear(run: str, frame_name: str, sim_state: dict):
    paths, names = _load_run_frames(run)
    if frame_name not in names: return None, [], sim_state
    fi = names.index(frame_name); sim_state[fi] = []
    im = cv.imread(paths[fi], cv.IMREAD_COLOR)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB), [], sim_state

def sim_save(run: str, sim_state: dict, label: str, score: float):
    if not run: return "Upload images first.", ""
    dets=[]
    for i, boxes in (sim_state or {}).items():
        for b in boxes:
            dets.append({"image_index": int(i), "bbox":[int(b[0]),int(b[1]),int(b[2]),int(b[3])],
                         "label": label or "manual", "score": float(score)})
    det_json = os.path.join(run, "detections.json")
    with open(det_json, "w") as f: json.dump(dets, f, indent=2)
    return f"Saved {len(dets)} boxes to detections.json", det_json

def friendly_summary(run: str):
    if not run: return "Upload images first.", []
    det_json = os.path.join(run, "detections.json")
    if not os.path.exists(det_json): return "No detections yet.", []
    dets = json.load(open(det_json,"r"))
    by_cls: Dict[str,int] = {}
    for d in dets:
        by_cls[d.get("label","fault")] = by_cls.get(d.get("label","fault"), 0) + 1
    header = [f"Detected {len(dets)} faults"]
    if by_cls:
        header.append("Classes: " + ", ".join([f"{k}: {v}" for k,v in sorted(by_cls.items())]))
    imgs_dir = os.path.join(run, "images")
    files = _sorted_images(imgs_dir)
    rows=[]
    for d in dets[:300]:
        idx = d["image_index"]
        img_name = os.path.basename(files[idx]) if 0<=idx<len(files) else f"IMG{idx}"
        rows.append([img_name, d.get("label","fault"),
                     _severity(float(d.get("score",1.0))), "Analysis placeholder"])
    return "\n".join(header), rows

def pdf_report(run: str) -> Optional[str]:
    if not run: return None
    det_json = os.path.join(run, "detections.json")
    if not os.path.exists(det_json): return None
    detections = json.load(open(det_json,"r"))
    imgs_dir = os.path.join(run, "images")
    files = _sorted_images(imgs_dir)
    pages=[]
    for i, fp in enumerate(files):
        im = Image.open(fp).convert("RGB")
        draw = ImageDraw.Draw(im)
        boxes = [d for d in detections if d["image_index"]==i]
        for d in boxes:
            x1,y1,x2,y2 = d["bbox"]
            draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=4)
            draw.text((x1+8,max(5,y1-24)), f"{d.get('label','fault')} {float(d.get('score',1.0)):.2f}", fill=(0,255,0))
        draw.text((10, 10), f"Analysis placeholder: {os.path.basename(fp)}", fill=(255,0,0))
        pages.append(im)
    if not pages: return None
    out_pdf = os.path.join(run, "report.pdf")
    pages[0].save(out_pdf, "PDF", resolution=100.0, save_all=True, append_images=pages[1:])
    return out_pdf

def do_stitch_and_project(run: str, reuse: bool = True):
    if not run: return "", "Upload images first.", "", ""
    imgs_dir = os.path.join(run, "images")
    files = _sorted_images(imgs_dir)
    if not files: return "", "No images.", "", ""
    pano = os.path.join(run, "pano.png")
    det_json = os.path.join(run, "detections.json")
    if not os.path.exists(det_json): return "", "No detections (run detection or simulation).", "", ""
    overlay = _overlay_path(pano)
    if reuse and os.path.exists(overlay) and os.path.exists(pano) and os.path.exists(_index_path(pano)):
        return overlay, "Using cached panorama and overlay.", overlay, pano
    try:
        stitcher = Stitcher(imgs_dir=imgs_dir, max_kpts=2000, yaw_align=True,
                            exposure=False, no_seams=False, num_bands=3, orb_fallback=False)
        pano_img, meta, exifs = stitcher.run()
        gmap = GeoMapper(align='yaw')
        phi_deg, A_world, epsg = gmap.choose_rotation(meta, exifs)
        pano_oriented, A_adj, epsg_out, world_json_path, M_pano, _ = gmap.orient_and_world(
            pano_img, phi_deg, A_world, epsg, pano
        )
        ok = cv.imwrite(pano, pano_oriented)
        if not ok: raise RuntimeError("Failed to save panorama")
        Hs = [np.array(H, np.float64) for H in meta["Hs_shift"]]
        if M_pano is None: M_pano = np.eye(3, dtype=np.float64)
        Hs_out = [(np.array(M_pano, np.float64) @ H).tolist() for H in Hs]
        index = {"mosaic_size":[int(pano_oriented.shape[1]), int(pano_oriented.shape[0])],
                 "H_proc_to_pano": Hs_out, "frames": meta.get("frame_models", [])}
        with open(_index_path(pano), "w") as f: json.dump(index, f, indent=2)
    except Exception as e:
        return "", f"Stitch failed: {e}", "", ""
    try:
        idx = json.load(open(_index_path(pano),"r"))
        dets = json.load(open(det_json,"r"))
        pj = PanoBoxProjector(idx, pano_bgr=cv.imread(pano, cv.IMREAD_COLOR))
        boxes_on_pano = pj.project(dets)
        pj.write_overlay(overlay)
        with open(_boxes_path(pano), "w") as f: json.dump(boxes_on_pano, f, indent=2)
        return overlay, f"Projected {len(boxes_on_pano)} boxes.", overlay, pano
    except Exception as e:
        return "", f"Projection failed: {e}", overlay if os.path.exists(overlay) else "", pano

def map_click(evt: gr.SelectData, run: str, open_maps: bool, zoom: int, use_gcps: bool):
    if not run: return "Upload images first."
    pano = os.path.join(run, "pano.png")
    gcps = _gcps_path(run) if use_gcps and os.path.exists(_gcps_path(run)) else None
    try:
        gl = PanoGeoLocalizer(pano, world_json=_world_path(pano),
                              gcps_json=gcps, boxes_json=_boxes_path(pano),
                              open_on_click=False, zoom=zoom)
    except Exception as e:
        return f"{e}"
    u, v = map(float, evt.index)
    lat, lon, E, N = gl.uv_to_latlon(u, v)
    url = _maps_url(lat, lon, zoom)
    if open_maps:
        try: webbrowser.open_new_tab(url)
        except: pass
    tail = f" · GCPs used: {_gcps_count(gcps)}" if gcps else " · GCPs: off"
    return f"lat/lon: `{lat:.7f}, {lon:.7f}`{tail}  ·  [Open in Google Maps]({url})"

def faults_table(run: str):
    if not run: return [], "Upload images first.", ""
    det_json = os.path.join(run, "detections.json")
    if not os.path.exists(det_json): return [], "No detections.", ""
    dets = json.load(open(det_json,"r"))
    imgs_dir = os.path.join(run, "images")
    files = _sorted_images(imgs_dir)
    rows=[]
    for d in dets:
        idx = d["image_index"]
        img_name = os.path.basename(files[idx]) if 0<=idx<len(files) else f"IMG{idx}"
        sev = _severity(float(d.get("score",1.0)))
        rows.append([img_name, d.get("label","fault"), f"{float(d.get('score',1.0)):.2f}", sev, "Analysis placeholder"])
    msg = f"{len(rows)} detections."
    first_img = files[dets[0]["image_index"]] if dets else ""
    return rows, msg, first_img

def build_fault_index(run: str):
    det_json = os.path.join(run, "detections.json")
    if not os.path.exists(det_json): return [], {}
    dets = json.load(open(det_json, "r"))
    imgs_dir = os.path.join(run, "images")
    files = _sorted_images(imgs_dir)
    index = {}
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
        x1,y1,x2,y2 = d["bbox"]
        cv.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3,cv.LINE_AA)
        cv.putText(im,f"{d.get('label','fault')} {float(d.get('score',1.0)):.2f}",
                   (x1,y1-6), cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv.LINE_AA)
        rows.append([os.path.basename(files[idx]), d.get("label","fault"),
                     f"{float(d.get('score',1.0)):.2f}", _severity(float(d.get('score',1.0))), "Analysis placeholder"])
    return cv.cvtColor(im, cv.COLOR_BGR2RGB), rows

def nav_data_loading(): return (gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
def nav_summary():      return (gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False))
def nav_map():          return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False))
def nav_faults():       return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
def nav_map_show(pano_overlay):
    return (*nav_map(), pano_overlay if pano_overlay else None)

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="pink",
    neutral_hue="slate",
).set(body_background_fill="#0f1220", body_text_color="#e7e9f1")

with gr.Blocks(title="Enevo — Solar Inspection", theme=theme, css="""
*{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}
.header{background:linear-gradient(90deg,#6d28d9 0%,#2563eb 50%,#22d3ee 100%);padding:18px 20px;border-radius:16px;color:white;margin-bottom:12px}
.header h1{font-size:22px;margin:0;font-weight:700;letter-spacing:.2px}
.section-title{opacity:.9;margin:4px 0 12px 0}
button{border-radius:12px}
""") as demo:
    gr.HTML('<div class="header"><h1>Enevo — Solar Inspection</h1></div>')

    run_state   = gr.State("")
    gcps_state  = gr.State("")
    click_state = gr.State({})
    sim_state   = gr.State({})
    faults_state = gr.State({"names": [], "index": {}, "files": []})
    current_fault_idx = gr.State(0)
    pano_overlay_state = gr.State("")
    pano_png_state = gr.State("")

    page_dl     = gr.Group(visible=True)
    with page_dl:
        gr.Markdown("### Data Loading", elem_classes=["section-title"])
        with gr.Row():
            uploads   = gr.File(file_count="multiple", file_types=["image"], label="Images")
            gcps_file = gr.File(file_types=[".json"], label="GCPs (optional)")
        with gr.Row():
            simulate_ck = gr.Checkbox(value=False, label="Simulation (manual boxes)")
            y_weights   = gr.Textbox(value="", label="YOLO weights (.pt/.pth)")
            y_data      = gr.Textbox(value="", label="data.yaml (optional)")
            y_classes   = gr.Textbox(value="", label="Classes")
        with gr.Row():
            y_imgsz = gr.Slider(320, 1536, value=640, step=32, label="imgsz")
            y_conf  = gr.Slider(0.05, 0.9, value=0.25, step=0.01, label="conf")
            y_iou   = gr.Slider(0.05, 0.9, value=0.45, step=0.01, label="iou")
            y_maxd  = gr.Slider(50, 1000, value=300, step=10, label="max_det")
        btn_copy = gr.Button("Load Files")
        log_copy = gr.Markdown()
        btn_gcps = gr.Button("Save GCPs")
        gcps_info = gr.Markdown()
        btn_detect = gr.Button("Run Detector / Enable Simulation")
        det_status = gr.Markdown()
        det_json_path = gr.Textbox(visible=False)
        btn_go_summary = gr.Button("Go to Summary", interactive=False)

        def _copy(files):
            run, log = dl_validate_and_copy(files)
            return run, log
        btn_copy.click(_copy, inputs=[uploads], outputs=[run_state, log_copy])
        btn_gcps.click(dl_attach_gcps, inputs=[run_state, gcps_file], outputs=[gcps_state, gcps_info])

        def _run_det_or_sim(run, sim, w, data, classes, imgsz, conf, iou, max_det):
            msg, detp = dl_run_detector_or_sim(run, sim, w, data, classes, int(imgsz), float(conf), float(iou), int(max_det))
            return msg, detp
        btn_detect.click(_run_det_or_sim,
                         inputs=[run_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
                         outputs=[det_status, det_json_path])

        sim_panel = gr.Group(visible=False)
        with sim_panel:
            gr.Markdown("### Simulation Annotator", elem_classes=["section-title"])
            with gr.Row():
                frame_dd = gr.Dropdown(choices=[], value=None, label="Frame")
                sim_img  = gr.Image(label="Annotate", interactive=True, height=560)
            sim_help = gr.Markdown()
            btn_init = gr.Button("Init")
            btn_prev = gr.Button("Prev"); btn_next = gr.Button("Next")
            with gr.Row():
                btn_undo = gr.Button("Undo"); btn_clear = gr.Button("Clear")
            with gr.Row():
                sim_label = gr.Textbox(value="manual", label="Label")
                sim_score = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Score")
                btn_save  = gr.Button("Save detections.json")
            sim_save_info = gr.Markdown()

            def _toggle_sim(vis): return gr.update(visible=bool(vis))
            simulate_ck.change(_toggle_sim, inputs=[simulate_ck], outputs=[sim_panel])

            btn_init.click(sim_init, inputs=[run_state], outputs=[frame_dd, sim_img, gr.State(), sim_help])

            def _prev(run, name):
                paths, names = _load_run_frames(run)
                if not names: return gr.update(), sim_state
                j = 0 if name not in names else max(0, names.index(name)-1)
                img, _ = sim_change_frame(run, names[j], sim_state.value or {})
                return gr.update(value=names[j]), img
            def _next(run, name):
                paths, names = _load_run_frames(run)
                if not names: return gr.update(), sim_state
                j = 0 if name not in names else min(len(names)-1, names.index(name)+1)
                img, _ = sim_change_frame(run, names[j], sim_state.value or {})
                return gr.update(value=names[j]), img
            btn_prev.click(_prev, inputs=[run_state, frame_dd], outputs=[frame_dd, sim_img])
            btn_next.click(_next, inputs=[run_state, frame_dd], outputs=[frame_dd, sim_img])

            sim_img.select(sim_click, inputs=[run_state, frame_dd, click_state, sim_state],
                           outputs=[sim_img, click_state, sim_state])
            btn_undo.click(sim_undo, inputs=[run_state, frame_dd, sim_state], outputs=[sim_img, gr.State(), sim_state])
            btn_clear.click(sim_clear, inputs=[run_state, frame_dd, sim_state], outputs=[sim_img, gr.State(), sim_state])

            def _frame_changed(run, name):
                img, _ = sim_change_frame(run, name, sim_state.value or {})
                return img
            frame_dd.change(_frame_changed, inputs=[run_state, frame_dd], outputs=[sim_img])

            btn_save.click(sim_save, inputs=[run_state, sim_state, sim_label, sim_score],
                           outputs=[sim_save_info, det_json_path])

        def _enable_summary_button(p): return gr.update(interactive=bool(p))
        btn_detect.click(_run_det_or_sim,
                         inputs=[run_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
                         outputs=[det_status, det_json_path]
                        ).then(_enable_summary_button, inputs=[det_json_path], outputs=[btn_go_summary])
        btn_save.click(sim_save, inputs=[run_state, sim_state, sim_label, sim_score],
                       outputs=[sim_save_info, det_json_path]
                      ).then(_enable_summary_button, inputs=[det_json_path], outputs=[btn_go_summary])

    page_summary = gr.Group(visible=False)
    with page_summary:
        gr.Markdown("### Summary", elem_classes=["section-title"])
        btn_refresh = gr.Button("Refresh")
        summary_md = gr.Markdown()
        table = gr.Dataframe(headers=["Image", "Class", "Severity", "Analysis"], row_count=10, wrap=True)
        with gr.Row():
            btn_to_map   = gr.Button("Show Map")
            btn_to_fault = gr.Button("Show Faults")
        btn_pdf   = gr.Button("Download Report (PDF)")
        pdf_file  = gr.File(label="Report", interactive=False)

        def _summ(run):
            txt, rows = friendly_summary(run)
            return txt, rows
        btn_refresh.click(_summ, inputs=[run_state], outputs=[summary_md, table])

        def _pdf(run):
            path = pdf_report(run)
            return path if path and os.path.exists(path) else None
        btn_pdf.click(_pdf, inputs=[run_state], outputs=[pdf_file])

    page_map = gr.Group(visible=False)
    with page_map:
        gr.Markdown("### Map", elem_classes=["section-title"])
        btn_stitch = gr.Button("Stitch & Project")
        pano_map   = gr.Image(type="filepath", label="", interactive=True, height=640)
        map_info   = gr.Markdown()
        with gr.Row():
            cb_open_maps = gr.Checkbox(value=False, label="Open Google Maps on click")
            cb_use_gcps  = gr.Checkbox(value=True,  label="Use GCPs")
            zoom_slider  = gr.Slider(10, 22, value=20, step=1, label="Zoom")

        def _stitch(run):
            img_val, msg, overlay, pano = do_stitch_and_project(run, reuse=True)
            return img_val, msg, overlay, pano
        btn_stitch.click(_stitch, inputs=[run_state],
                         outputs=[pano_map, map_info, pano_overlay_state, pano_png_state])
        pano_map.select(map_click, inputs=[run_state, cb_open_maps, zoom_slider, cb_use_gcps], outputs=[map_info])

    page_faults = gr.Group(visible=False)
    with page_faults:
        gr.Markdown("### Faults", elem_classes=["section-title"])
        btn_load_faults = gr.Button("Load")
        faults_md = gr.Markdown()
        faults_tbl = gr.Dataframe(headers=["Image","Class","Score","Severity","Analysis"], row_count=10, wrap=True)
        with gr.Row():
            img_name_dd = gr.Dropdown(choices=[], label="Image")
            btn_prev_img = gr.Button("Prev")
            btn_next_img = gr.Button("Next")
        img_preview = gr.Image(label="", interactive=False, height=560)

        def _load_faults(run):
            names, det_index, files = build_fault_index(run)
            if names:
                rows_all, msg, _ = faults_table(run)
                first_idx_val = names[0][1]
                img, rows_first = render_fault_image(files, first_idx_val, det_index)
                return (msg, rows_all,
                        gr.update(choices=[n for n,_ in names], value=names[0][0]),
                        img, {"names": names, "index": det_index, "files": files}, first_idx_val)
            return ("No detections.", [], gr.update(choices=[], value=None), None,
                    {"names": [], "index": {}, "files": []}, 0)

        btn_load_faults.click(_load_faults, inputs=[run_state],
                              outputs=[faults_md, faults_tbl, img_name_dd, img_preview, faults_state, current_fault_idx])

        def _select_name(name, state):
            names = state.get("names", [])
            files = state.get("files", [])
            det_index = state.get("index", {})
            if not names: return None, 0
            map_names = {n:i for n,i in names}
            idx = map_names.get(name, names[0][1])
            img, rows = render_fault_image(files, idx, det_index)
            return img, idx

        img_name_dd.change(_select_name, inputs=[img_name_dd, faults_state], outputs=[img_preview, current_fault_idx])

        def _prev_img(state, cur_idx):
            names = state.get("names", [])
            files = state.get("files", [])
            det_index = state.get("index", {})
            if not names: return gr.update(), cur_idx, None
            order = [i for _,i in names]
            if cur_idx not in order: cur_idx = order[0]
            j = max(0, order.index(cur_idx)-1)
            new_idx = order[j]
            new_name = [n for n,i in names if i==new_idx][0]
            img, rows = render_fault_image(files, new_idx, det_index)
            return gr.update(value=new_name), new_idx, img

        def _next_img(state, cur_idx):
            names = state.get("names", [])
            files = state.get("files", [])
            det_index = state.get("index", {})
            if not names: return gr.update(), cur_idx, None
            order = [i for _,i in names]
            if cur_idx not in order: cur_idx = order[0]
            j = min(len(order)-1, order.index(cur_idx)+1)
            new_idx = order[j]
            new_name = [n for n,i in names if i==new_idx][0]
            img, rows = render_fault_image(files, new_idx, det_index)
            return gr.update(value=new_name), new_idx, img

        btn_prev_img.click(_prev_img, inputs=[faults_state, current_fault_idx], outputs=[img_name_dd, current_fault_idx, img_preview])
        btn_next_img.click(_next_img, inputs=[faults_state, current_fault_idx], outputs=[img_name_dd, current_fault_idx, img_preview])

    def nav_data():  return nav_data_loading()
    def nav_sum():   return nav_summary()
    def nav_map_go(pano_overlay): return nav_map_show(pano_overlay)
    def nav_fault(): return nav_faults()

    btn_go_summary.click(nav_sum, inputs=[], outputs=[page_dl, page_summary, page_map, page_faults])
    btn_to_map.click(nav_map_go, inputs=[pano_overlay_state], outputs=[page_dl, page_summary, page_map, page_faults, pano_map])
    btn_to_fault.click(nav_fault, inputs=[], outputs=[page_dl, page_summary, page_map, page_faults])

    with page_map:
        btn_back_sum = gr.Button("Back to Summary")
        btn_goto_fault = gr.Button("Go to Faults")
    btn_back_sum.click(nav_sum, inputs=[], outputs=[page_dl, page_summary, page_map, page_faults])
    btn_goto_fault.click(nav_fault, inputs=[], outputs=[page_dl, page_summary, page_map, page_faults])

    with page_faults:
        btn_back_sum2 = gr.Button("Back to Summary")
        btn_to_map2   = gr.Button("Go to Map")
    btn_back_sum2.click(nav_sum, inputs=[], outputs=[page_dl, page_summary, page_map, page_faults])
    btn_to_map2.click(nav_map_go, inputs=[pano_overlay_state], outputs=[page_dl, page_summary, page_map, page_faults, pano_map])

if __name__ == "__main__":
    demo.launch()
