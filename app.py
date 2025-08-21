#!/usr/bin/env python3
import os, json, shutil, time, uuid, webbrowser
from typing import List, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import cv2 as cv

# --- wire in your local core modules ---
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

# Step 1
from core.stitcher import Stitcher
from core.mapper import GeoMapper
# Step 2
from core.detector import YOLODetector
from core.simulator import DetectionSimulator
from core.box_projector import PanoBoxProjector
# Step 3
from core.geolocalizer import PanoGeoLocalizer


# ========= app storage =========
RUNS_DIR = os.path.join(HERE, "app_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

def _new_run_dir() -> str:
    d = os.path.join(RUNS_DIR, time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "det_vis"), exist_ok=True)
    return d

def _overlay_path(out_png: str) -> str:
    return os.path.splitext(out_png)[0] + ".overlay.png"

def _boxes_path(out_png: str) -> str:
    return os.path.splitext(out_png)[0] + ".boxes.json"

def _world_path(out_png: str) -> str:
    return os.path.splitext(out_png)[0] + ".world.json"

def _index_path(out_png: str) -> str:
    return os.path.splitext(out_png)[0] + ".index.json"


# ========= STEP 1 (Data loading + Stitch) =========
def ui_data_load(files: List[gr.File]) -> Tuple[str, str, str, dict]:
    """
    Copy uploads to a run folder and run Step 1 (stitch + world + index).
    Returns: (run_dir, pano, status, summary_dict)
    """
    if not files:
        return "", "", "⚠️ Please upload images first.", {}
    run = _new_run_dir()
    imgs_dir = os.path.join(run, "images")
    for f in files:
        # Gradio gives a tempfile path; we keep original names if possible
        name = os.path.basename(getattr(f, "name", "img.jpg"))
        # keep only image files
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")):
            continue
        shutil.copy(f.name, os.path.join(imgs_dir, name))

    pano_path = os.path.join(run, "pano.png")
    try:
        # Step 1
        stitcher = Stitcher(
            imgs_dir=imgs_dir,
            max_kpts=2000,
            yaw_align=True,
            exposure=False,
            no_seams=False,
            num_bands=3,
            orb_fallback=False
        )
        pano, meta, exifs = stitcher.run()

        gmap = GeoMapper(align='yaw')
        phi_deg, A_world, epsg = gmap.choose_rotation(meta, exifs)
        pano_oriented, A_adj, epsg_out, world_json_path, M_pano, _ = gmap.orient_and_world(
            pano, phi_deg, A_world, epsg, pano_path
        )
        ok = cv.imwrite(pano_path, pano_oriented)
        if not ok:
            raise RuntimeError("Failed to save panorama.")

        # save index
        from pathlib import Path
        Hs = [np.array(H, np.float64) for H in meta["Hs_shift"]]
        if M_pano is None:
            M_pano = np.eye(3, dtype=np.float64)
        Hs_out = [(np.array(M_pano, np.float64) @ H).tolist() for H in Hs]
        idx = {
            "mosaic_size": [int(pano_oriented.shape[1]), int(pano_oriented.shape[0])],
            "H_proc_to_pano": Hs_out,
            "frames": meta.get("frame_models", [])
        }
        with open(_index_path(pano_path), "w") as f:
            json.dump(idx, f, indent=2)

        # small summary
        gps_ok = os.path.exists(_world_path(pano_path))
        summary = {
            "run_dir": run,
            "images": len(meta.get("frame_models", [])),
            "pano": pano_path,
            "world_json": gps_ok,
            "index_json": True
        }
        return run, pano_path, "✅ Stitch complete.", summary

    except Exception as e:
        return run, "", f"❌ Stitch failed: {e}", {}


# ========= STEP 2 (Detect or Simulate + Project) =========
def ui_step2_detect(run_dir: str, mode: str, weights: str, data_yaml: str,
                    classes: str, imgsz: int, conf: float, iou: float, max_det: int) -> Tuple[str, str, dict]:
    """
    Run detector OR open simulator, then project detections to pano.
    Returns: (overlay_png, boxes_json_path, stats_dict)
    """
    if not run_dir or not os.path.isdir(run_dir):
        return "", "", {"error": "Invalid run. Load data first."}

    pano = os.path.join(run_dir, "pano.png")
    idx_path = _index_path(pano)
    if not (os.path.exists(pano) and os.path.exists(idx_path)):
        return "", "", {"error": "Missing pano or index. Run Step 1 first."}

    idx = json.load(open(idx_path, "r"))
    det_vis_dir = os.path.join(run_dir, "det_vis")

    # 2a) detections
    if mode == "detect":
        if not weights:
            return "", "", {"error": "Please provide YOLO weights (.pt/.pth)."}
        det = YOLODetector(weights=weights, imgsz=imgsz, conf=conf, iou=iou,
                           max_det=max_det, classes=classes, data_yaml=data_yaml)
        dets = det.run_on_index(idx, limit=None, save_vis_dir=det_vis_dir)
    else:
        # manual simulator in a notebook/server isn’t ideal; instead, accept empty list here
        dets = []  # user can pre-create detections.json and drop into run_dir if needed

    # 2b) project to pano
    pj = PanoBoxProjector(idx, pano_bgr=cv.imread(pano, cv.IMREAD_COLOR))
    boxes_on_pano = pj.project(dets)
    pj.write_overlay(_overlay_path(pano))
    with open(_boxes_path(pano), "w") as f:
        json.dump(boxes_on_pano, f, indent=2)

    stats = {
        "detections": len(dets),
        "projected": len(boxes_on_pano),
        "overlay": _overlay_path(pano),
        "boxes_json": _boxes_path(pano),
        "det_vis_dir": det_vis_dir
    }
    return _overlay_path(pano), _boxes_path(pano), stats


# ========= STEP 3 (Geolocalize clicks / boxes) =========
def _maps_url(lat: float, lon: float, zoom: int = 20) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.7f}%2C{lon:.7f}&zoom={zoom}"

def ui_map_load(run_dir: str) -> Tuple[str, List[Dict], str]:
    """Return pano overlay + boxes list + info text."""
    if not run_dir:
        return "", [], "⚠️ Load data first."
    pano = os.path.join(run_dir, "pano.png")
    boxes_path = _boxes_path(pano)
    overlay = _overlay_path(pano) if os.path.exists(_overlay_path(pano)) else pano
    boxes = json.load(open(boxes_path, "r")) if os.path.exists(boxes_path) else []
    return overlay, boxes, f"Loaded {len(boxes)} projected boxes."

def ui_map_click(evt: gr.SelectData, run_dir: str, open_maps: bool, zoom: int) -> str:
    """Click anywhere on pano → lat/lon using PanoGeoLocalizer."""
    if not run_dir:
        return "⚠️ Load data first."
    pano = os.path.join(run_dir, "pano.png")
    try:
        gl = PanoGeoLocalizer(pano, world_json=_world_path(pano),
                              gcps_json=None, boxes_json=_boxes_path(pano),
                              open_on_click=False, zoom=zoom)
    except Exception as e:
        return f"❌ {e}"

    u, v = map(float, evt.index)  # (x,y) from displayed image
    lat, lon, E, N = gl.uv_to_latlon(u, v)
    url = _maps_url(lat, lon, zoom)
    if open_maps:
        try: webbrowser.open_new_tab(url)
        except: pass
    return f"**lat, lon** = `{lat:.7f}, {lon:.7f}`  ·  **E,N** = `{E:.2f}, {N:.2f}`  ·  [Open in Google Maps]({url})"

def ui_first_detection_open(run_dir: str, open_maps: bool, zoom: int) -> str:
    """Open Google Maps at the first box center (quick sanity button)."""
    if not run_dir:
        return "⚠️ Load data first."
    pano = os.path.join(run_dir, "pano.png")
    boxes_path = _boxes_path(pano)
    if not os.path.exists(boxes_path):
        return "⚠️ No boxes (run Step 2)."
    boxes = json.load(open(boxes_path, "r"))
    if not boxes:
        return "⚠️ No boxes to open."
    b = boxes[0]
    q = np.asarray(b["quad"], np.float64)
    cx, cy = float(q[:,0].mean()), float(q[:,1].mean())

    gl = PanoGeoLocalizer(pano, world_json=_world_path(pano),
                          gcps_json=None, boxes_json=boxes_path,
                          open_on_click=False, zoom=zoom)
    lat, lon, E, N = gl.uv_to_latlon(cx, cy)
    url = _maps_url(lat, lon, zoom)
    if open_maps:
        try: webbrowser.open_new_tab(url)
        except: pass
    return f"Box #{b.get('id','0')} → **lat, lon** = `{lat:.7f}, {lon:.7f}`  ·  [Open in Google Maps]({url})"


# ========= Gradio UI =========
with gr.Blocks(title="Enevo Demo App", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🌞 Enevo Demo App\n*Stitch → Detect/Simulate → Project → Geolocalize*")

    # shared state across tabs
    run_state = gr.State("")     # run_dir path
    summary_state = gr.State({}) # dict summary from step1
    step2stats_state = gr.State({})

    with gr.Tabs():

        # -------------------- Data Loading --------------------
        with gr.Tab("Data Loading"):
            gr.Markdown("Upload your **drone images** and press **Process & Stitch**. "
                        "All data is validated and processed before moving on.")
            uploads = gr.File(file_count="multiple", file_types=["image"], label="Upload images")
            btn_stitch = gr.Button("Process & Stitch", variant="primary")
            out_pano = gr.Image(label="Panorama preview", interactive=False)
            status = gr.Markdown()
            summary_json = gr.JSON(label="Summary")

            def _run_stitch(files):
                run, pano, msg, summary = ui_data_load(files)
                return run, summary, gr.update(value=pano, visible=bool(pano)), msg

            btn_stitch.click(_run_stitch, inputs=[uploads],
                             outputs=[run_state, summary_state, out_pano, status])

        # -------------------- Dashboard #1 --------------------
        with gr.Tab("Dashboard #1"):
            gr.Markdown("### Summary\nOptions for detailed info. You can download results or proceed.")
            show_summary = gr.Button("Refresh Summary")
            sum_out = gr.JSON(label="Summary")
            dl_world = gr.File(label="pano.world.json")
            dl_index = gr.File(label="pano.index.json")

            def _refresh_summary(run):
                if not run:
                    return {}, None, None
                pano = os.path.join(run, "pano.png")
                wp, ip = _world_path(pano), _index_path(pano)
                wfile = wp if os.path.exists(wp) else None
                ifile = ip if os.path.exists(ip) else None
                return json.load(open(_index_path(pano))) if os.path.exists(ip) else {}, wfile, ifile

            show_summary.click(_refresh_summary, inputs=[run_state], outputs=[sum_out, dl_world, dl_index])

            gr.Markdown("**Next:** use tabs to open **Dashboard #2 (Map)** or **Dashboard #3 (Faults)**.")

        # -------------------- Dashboard #2 (Map) --------------------
        with gr.Tab("Dashboard #2"):
            gr.Markdown("### Map (Geolocation) + Overview (Stitching)")
            row = gr.Row()
            with row:
                pano_map = gr.Image(type="filepath", label="Panorama (with overlay if available)", interactive=True)
                boxes_view = gr.JSON(label="Projected boxes (pano.quads)")
            info_click = gr.Markdown()
            with gr.Row():
                btn_load_map = gr.Button("Load Map/Overlay", variant="primary")
                cb_open_maps = gr.Checkbox(value=False, label="Open Google Maps on click")
                zoom_slider = gr.Slider(10, 22, value=20, step=1, label="Maps zoom")
                btn_open_first = gr.Button("Open first detection in Maps")

            def _load_map(run):
                overlay, boxes, msg = ui_map_load(run)
                return overlay, boxes, msg

            btn_load_map.click(_load_map, inputs=[run_state],
                               outputs=[pano_map, boxes_view, info_click])

            # click anywhere on pano => lat/lon
            pano_map.select(ui_map_click, inputs=[run_state, cb_open_maps, zoom_slider], outputs=[info_click])
            btn_open_first.click(ui_first_detection_open, inputs=[run_state, cb_open_maps, zoom_slider], outputs=[info_click])

        # -------------------- Dashboard #3 (Faults) --------------------
        with gr.Tab("Dashboard #3"):
            gr.Markdown("### Fault detection per image (detector or simulator)")
            with gr.Row():
                det_mode = gr.Radio(choices=["detect", "simulate"], value="detect", label="Mode")
                y_weights = gr.Textbox(value="", label="YOLO weights (.pt/.pth)")
                y_data = gr.Textbox(value="", label="data.yaml (optional names override)")
            with gr.Row():
                y_classes = gr.Textbox(value="", label="Classes (e.g. '0,1' or 'fault,crack')")
                y_imgsz = gr.Slider(320, 1536, value=640, step=32, label="imgsz")
                y_conf  = gr.Slider(0.05, 0.9, value=0.25, step=0.01, label="conf")
                y_iou   = gr.Slider(0.05, 0.9, value=0.45, step=0.01, label="iou")
                y_maxd  = gr.Slider(50, 1000, value=300, step=10, label="max_det")
            btn_run2 = gr.Button("Run Step 2 (Detect/Simulate + Project)", variant="primary")

            det_overlay = gr.Image(label="Pano overlay", interactive=False)
            det_boxes_json = gr.File(label="Projected boxes JSON")
            det_stats = gr.JSON(label="Stats")
            det_gallery = gr.Gallery(label="Per-image detections (if saved)").style(grid=[3], height="auto")

            def _run_step2(run, mode, weights, data, classes, imgsz, conf, iou, max_det):
                overlay, boxes_json, stats = ui_step2_detect(run, mode, weights, data, classes, int(imgsz), float(conf), float(iou), int(max_det))
                # build a simple gallery from det_vis dir
                gallery = []
                if stats.get("det_vis_dir") and os.path.isdir(stats["det_vis_dir"]):
                    for fn in sorted(os.listdir(stats["det_vis_dir"]))[:60]:
                        if fn.lower().endswith((".jpg",".png",".jpeg",".webp")):
                            gallery.append(os.path.join(stats["det_vis_dir"], fn))
                return overlay or None, boxes_json or None, stats, gallery

            btn_run2.click(_run_step2, inputs=[run_state, det_mode, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
                           outputs=[det_overlay, det_boxes_json, det_stats, det_gallery])

    gr.Markdown("---\nTips: Use **Dashboard #2** to click anywhere on the pano and get **lat/lon** (toggle “Open Google Maps on click”).")

if __name__ == "__main__":
    demo.launch()
