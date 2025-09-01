import os, sys
import tempfile, shutil
import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from app_utils import validate_and_copy, attach_gcps
from services import (
    run_detector_or_sim, sim_init, sim_change_frame, sim_click, sim_undo, sim_clear, sim_save,
    friendly_summary, pdf_report, do_stitch_and_project, map_click,
    faults_table, build_fault_index, render_fault_image, build_fault_cards,
)

# ---- Theme / CSS ----
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="pink",
    neutral_hue="slate",
).set(body_background_fill="#0f1220", body_text_color="#e7e9f1")

CSS = """
*{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}
.header{background:linear-gradient(90deg,#6d28d9 0%,#2563eb 50%,#22d3ee 100%);padding:18px 20px;border-radius:16px;color:white;margin-bottom:12px}
.header h1{font-size:22px;margin:0;font-weight:700;letter-spacing:.2px}
.section-title{opacity:.9;margin:4px 0 12px 0}
button{border-radius:12px}

/* Bottom nav buttons */
.nav-row{display:flex;gap:10px;justify-content:flex-end;margin-top:12px}
.navbtn{
  background:transparent; color:#22d3ee; font-weight:700; border-radius:12px;
  box-shadow: 0 0 0 2px #22d3ee inset, 0 0 16px rgba(34,211,238,.35);
  border: none; padding:10px 14px;
}
.navbtn:hover{
  box-shadow: 0 0 0 2px #22d3ee inset, 0 0 24px rgba(34,211,238,.55);
  transform: translateY(-1px);
}

/* Faults scroller */
.faults-scroller{
  height: 560px;
  overflow-y: auto;
  padding-right: 8px;
}
.faults-scroller::-webkit-scrollbar{ width: 10px; }
.faults-scroller::-webkit-scrollbar-thumb{ background: #22d3ee88; border-radius: 8px; }
.faults-scroller::-webkit-scrollbar-track{ background: transparent; }
"""

def build_ui():
    with gr.Blocks(title="Solar Inspection", theme=theme, css=CSS) as demo:
        gr.HTML('<div class="header"><h1>Solar Inspection</h1></div>')

        # ---------- Reactive state ----------
        run_state           = gr.State("")
        gcps_state          = gr.State("")
        click_state         = gr.State({})
        sim_state           = gr.State({})
        faults_state        = gr.State({"names": [], "index": {}, "files": []})
        current_fault_idx   = gr.State(0)
        pano_overlay_state  = gr.State("")
        pano_png_state      = gr.State("")
        det_tmp_state       = gr.State("")
        faults_loaded_state = gr.State(False)

        # ---------- Simple nav helpers ----------
        def show_data():    return (gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        def show_summary(): return (gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False))
        def show_map():     return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False))
        def show_faults():  return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

        # ---------- Declare pages ----------
        page_data    = gr.Group(visible=True)
        page_summary = gr.Group(visible=False)
        page_map     = gr.Group(visible=False)
        page_faults  = gr.Group(visible=False)

        # =================== PAGE: DATA ===================
        with page_data:
            gr.Markdown("### Data Loading", elem_classes=["section-title"])
            with gr.Row():
                uploads = gr.File(file_count="multiple", file_types=["image", ".json"], label="Images and GCPs.json")

            btn_copy = gr.Button("Load Files")
            log_copy = gr.Markdown()
            gcps_auto_info = gr.Markdown()

            # Hidden holder for detections path (used by Simulation Save)
            det_json_path = gr.Textbox(visible=False)

            def _filter_and_copy(files):
                files = files or []
                def _name(x):
                    try:
                        return getattr(x, "name", str(x))
                    except Exception:
                        return str(x)
                image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
                img_files = [f for f in files if _name(f).lower().endswith(image_exts)]
                return validate_and_copy(img_files)

            def _load_files_and_gcps(files):
                run, log = _filter_and_copy(files)
                files = files or []
                def _name(x):
                    try:
                        return getattr(x, "name", str(x))
                    except Exception:
                        return str(x)
                json_files = [f for f in files if _name(f).lower().endswith(".json")]
                if json_files and run:
                    g_state, g_info = attach_gcps(run, json_files[0])
                else:
                    g_state, g_info = "", ""
                return run, log, g_state, g_info

            btn_copy.click(_load_files_and_gcps, inputs=[uploads], outputs=[run_state, log_copy, gcps_state, gcps_auto_info])

            # Advanced settings (interface unchanged; no Save GCPs button)
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    simulate_ck = gr.Checkbox(value=False, label="Simulation (manual boxes)")
                    y_weights   = gr.Textbox(value="model.pt", label="YOLO weights (.pt/.pth)")
                    y_data      = gr.Textbox(value="", label="data.yaml (optional)")
                    y_classes   = gr.Textbox(value="", label="Classes")
                with gr.Row():
                    y_imgsz = gr.Slider(320, 1536, value=640, step=32, label="imgsz")
                    y_conf  = gr.Slider(0.05, 0.9, value=0.25, step=0.01, label="conf")
                    y_iou   = gr.Slider(0.05, 0.9, value=0.45, step=0.01, label="iou")
                    y_maxd  = gr.Slider(50, 1000, value=300, step=10, label="max_det")

                # Simulation subpanel
                with gr.Accordion("Simulation Annotator", open=False) as sim_panel:
                    with gr.Row():
                        frame_dd = gr.Dropdown(choices=[], value=None, label="Frame")
                        sim_img  = gr.Image(label="Annotate", interactive=True, height=560)
                    sim_help = gr.Markdown()
                    btn_init = gr.Button("Init")
                    with gr.Row():
                        btn_undo = gr.Button("Undo"); btn_clear = gr.Button("Clear")
                    with gr.Row():
                        sim_label = gr.Textbox(value="manual", label="Label")
                        sim_score = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Score")
                        btn_save  = gr.Button("Save detections.json")
                    sim_save_info = gr.Markdown()

                    def _toggle_sim(vis): return gr.update(open=bool(vis))
                    simulate_ck.change(_toggle_sim, inputs=[simulate_ck], outputs=[sim_panel])

                    btn_init.click(sim_init, inputs=[run_state], outputs=[frame_dd, sim_img, sim_state, sim_help])
                    sim_img.select(sim_click, inputs=[run_state, frame_dd, click_state, sim_state], outputs=[sim_img, click_state, sim_state])
                    btn_undo.click(sim_undo, inputs=[run_state, frame_dd, sim_state], outputs=[sim_img, sim_state])
                    btn_clear.click(sim_clear, inputs=[run_state, frame_dd, sim_state], outputs=[sim_img, sim_state])
                    frame_dd.change(lambda run, name: sim_change_frame(run, name, sim_state.value or {})[0],
                                    inputs=[run_state, frame_dd], outputs=[sim_img])
                    btn_save.click(sim_save, inputs=[run_state, sim_state, sim_label, sim_score], outputs=[sim_save_info, det_json_path])

            # Quick nav from Data -> Summary
            with gr.Row(elem_classes=["nav-row"]):
                btn_nav_summary_from_data = gr.Button("View Summary", elem_classes=["navbtn"])

        # =================== PAGE: MAP ===================
        with page_map:
            gr.Markdown("### Map", elem_classes=["section-title"])
            btn_stitch = gr.Button("Stitch & Project")

            pano_map = gr.Image(
                type="filepath", label="Panorama", interactive=True, height=640,
                sources=[], show_download_button=True, show_fullscreen_button=True
            )
            map_info = gr.Markdown()

            # Replaced zoom slider with Save Image
            with gr.Row():
                cb_open_maps = gr.Checkbox(value=False, label="Open Google Maps on click")
                cb_use_gcps  = gr.Checkbox(value=True,  label="Use GCPs")
                btn_save_img = gr.Button("Save Image")
                pano_file_dl = gr.File(label="Panorama (download)", interactive=False)

            # Save Image -> expose the pano path as a downloadable file
            def _save_pano(pano_path):
                if pano_path and os.path.isfile(pano_path):
                    return pano_path
                return None

            btn_save_img.click(_save_pano, inputs=[pano_png_state], outputs=[pano_file_dl])

            btn_stitch.click(
                lambda run: do_stitch_and_project(run, reuse=True),
                inputs=[run_state],
                outputs=[pano_map, map_info, pano_overlay_state, pano_png_state],
            )

            # Clicking on the panorama -> lat/lon + optional browser open (fixed zoom=20)
            pano_map.select(
                lambda run, open_maps, use_gcps: map_click(run, open_maps, 20, use_gcps),
                inputs=[run_state, cb_open_maps, cb_use_gcps],
                outputs=[map_info],
            )

            # Bottom nav
            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults]
                )
                btn_nav_summary_from_map = gr.Button("View Summary", elem_classes=["navbtn"])
                btn_nav_faults_from_map  = gr.Button("View Faults", elem_classes=["navbtn"])

        # =================== PAGE: SUMMARY ===================
        with page_summary:
            gr.Markdown("### Summary", elem_classes=["section-title"])
            btn_refresh   = gr.Button("Refresh")
            summary_md    = gr.Markdown()
            table         = gr.Dataframe(headers=["ImageIndex", "Class", "Severity", "Analysis"], row_count=10, wrap=True)
            btn_pdf       = gr.Button("Download Report (PDF)")
            pdf_file      = gr.File(label="Report", interactive=False)
            det_json_file = gr.File(label="Detections JSON", interactive=False)

            def _summary_and_det(run, tmp_path):
                md, rows = friendly_summary(run)  # returns 2 values
                # Prefer temp file if exists, otherwise run detections.json if present
                if tmp_path and os.path.isfile(tmp_path):
                    det_path = tmp_path
                else:
                    det_path = os.path.join(run or "", "detections.json") if run else ""
                    if not (det_path and os.path.isfile(det_path)):
                        det_path = None
                return md, rows, det_path

            btn_refresh.click(_summary_and_det, inputs=[run_state, det_tmp_state], outputs=[summary_md, table, det_json_file])
            btn_pdf.click(lambda run: pdf_report(run), inputs=[run_state], outputs=[pdf_file])

            # Bottom nav
            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults]
                )
                btn_nav_faults_from_summary = gr.Button("View Faults", elem_classes=["navbtn"])
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults]
                )

        # =================== PAGE: FAULTS ===================
        with page_faults:
            gr.Markdown("### Faults", elem_classes=["section-title"])
            btn_load_faults = gr.Button("Load")
            with gr.Group(elem_classes=["faults-scroller"]):
                cards = gr.Gallery(label="Fault Cards", columns=[3])

            def _load_faults(run):
                names, det_index, files = build_fault_index(run)
                if names:
                    gallery = build_fault_cards(run)
                    return (gallery, {"names": names, "index": det_index, "files": files}, True)
                return ([], {"names": [], "index": {}, "files": []}, True)

            btn_load_faults.click(_load_faults, inputs=[run_state],
                                  outputs=[cards, faults_state, faults_loaded_state])

            # Bottom nav
            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults]
                )
                btn_nav_summary_from_faults = gr.Button("View Summary", elem_classes=["navbtn"])
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults]
                )

        # ---- Nav helper that also refreshes Map when stitching finishes ----
        def _nav_summary(run, tmp_path, sim, w, data, classes, imgsz, conf, iou, max_det):
            """
            1) Navigate to Summary immediately;
            2) Ensure detections exist;
            3) Stitch; when done, push pano to the Map page (no extra clicks).
            """
            # Step 1: ensure detections exist (if simulate checked, respect it)
            det_json = os.path.join(run or "", "detections.json") if run else ""
            if not (det_json and os.path.isfile(det_json)):
                w = w or "model.pt"
                try:
                    _msg, detp_new = run_detector_or_sim(run, bool(sim), w, data, classes, int(imgsz), float(conf), float(iou), int(max_det))
                    faults_loaded_state.value = False  # invalidate cache on new dets
                    det_json = detp_new or det_json
                    if detp_new and os.path.isfile(detp_new):
                        fd, tpath = tempfile.mkstemp(suffix=".json", prefix="detections_")
                        os.close(fd)
                        shutil.copyfile(detp_new, tpath)
                        det_tmp_state.value = tpath
                except Exception:
                    pass

            # Step 2: prepare initial Summary payload using the same helper as Refresh
            v = show_summary()
            md, rows, detp = _summary_and_det(run, det_tmp_state.value or tmp_path)

            # Initial yield: update Summary immediately; don't touch Map yet
            yield (*v, md, rows, detp, gr.update(), gr.update())

            # Step 3: stitch synchronously; when done, push pano to Map
            try:
                img_out, msg, ov_abs, pano_abs = do_stitch_and_project(run, reuse=True)
                pano_overlay_state.value = ov_abs
                pano_png_state.value     = pano_abs
                # Second yield: keep Summary the same, but populate Map so it's ready
                yield (*v, md, rows, detp, img_out, msg)
            except Exception:
                yield (*v, md, rows, detp, gr.update(), gr.update())

        def _nav_faults(run, loaded):
            v = show_faults()
            if loaded:
                return (*v, gr.update(), gr.update(), loaded)
            gallery, st, loaded_out = _load_faults(run)
            return (*v, gallery, st, loaded_out)

        # ---- Wire nav buttons (also update Map components) ----
        btn_nav_summary_from_data.click(
            _nav_summary,
            inputs=[run_state, det_tmp_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
            outputs=[page_data, page_summary, page_map, page_faults, summary_md, table, det_json_file, pano_map, map_info]
        )
        btn_nav_summary_from_map.click(
            _nav_summary,
            inputs=[run_state, det_tmp_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
            outputs=[page_data, page_summary, page_map, page_faults, summary_md, table, det_json_file, pano_map, map_info]
        )
        btn_nav_faults_from_map.click(
            _nav_faults,
            inputs=[run_state, faults_loaded_state],
            outputs=[page_data, page_summary, page_map, page_faults, cards, faults_state, faults_loaded_state]
        )
        btn_nav_faults_from_summary.click(
            _nav_faults,
            inputs=[run_state, faults_loaded_state],
            outputs=[page_data, page_summary, page_map, page_faults, cards, faults_state, faults_loaded_state]
        )
        btn_nav_summary_from_faults.click(
            _nav_summary,
            inputs=[run_state, det_tmp_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
            outputs=[page_data, page_summary, page_map, page_faults, summary_md, table, det_json_file, pano_map, map_info]
        )

    return demo

if __name__ == "__main__":
    build_ui().launch()
