import os, sys, json, tempfile, shutil
import gradio as gr

# --- Project paths
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- App internals already in your project
from app_utils import validate_and_copy, attach_gcps
from services import (
    run_detector_or_sim, sim_init, sim_change_frame, sim_click, sim_undo, sim_clear, sim_save,
    friendly_summary, pdf_report, do_stitch_and_project, map_click,
    build_fault_index, build_fault_cards,
)

# --- Thermography (moved out of app.py)
from thermo_panel import (
    THERMO_OK, THERMO_IMPORT_ERR,
    run_modules,  # main entry called by the “Run Thermography” button
)

# --- Theme / CSS
theme = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="pink", neutral_hue="slate",
).set(body_background_fill="#0f1220", body_text_color="#e7e9f1")

CSS = """
*{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}
.header{background:linear-gradient(90deg,#6d28d9 0%,#2563eb 50%,#22d3ee 100%);padding:18px 20px;border-radius:16px;color:white;margin-bottom:12px}
.header h1{font-size:22px;margin:0;font-weight:700;letter-spacing:.2px}
.section-title{opacity:.9;margin:4px 0 12px 0}
button{border-radius:12px}
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
.faults-scroller{height:560px;overflow-y:auto;padding-right:8px}
.faults-scroller::-webkit-scrollbar{width:10px}
.faults-scroller::-webkit-scrollbar-thumb{background:#22d3ee88;border-radius:8px}
.faults-scroller::-webkit-scrollbar-track{background:transparent}
"""

# ======================================================================================
# UI ONLY FROM HERE
# ======================================================================================
def build_ui():
    with gr.Blocks(title="Solar Inspection", theme=theme, css=CSS) as demo:
        gr.HTML('<div class="header"><h1>Solar Inspection</h1></div>')

        # --- cross-page state
        run_state           = gr.State("")
        gcps_state          = gr.State("")
        click_state         = gr.State({})          # <-- keep a State, don't pass raw {}
        sim_state           = gr.State({})
        pano_overlay_state  = gr.State("")
        pano_png_state      = gr.State("")
        det_tmp_state       = gr.State("")
        faults_loaded_state = gr.State(False)
        faults_state        = gr.State({"names": [], "index": {}, "files": []})

        # panel-modules state
        modules_run_state   = gr.State("")
        modules_files_state = gr.State([])
        modules_workdir     = gr.State("")

        # visibility helpers
        def show_data():    return (gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        def show_summary(): return (gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        def show_map():     return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False))
        def show_faults():  return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False))
        def show_modules(): return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

        page_data    = gr.Group(visible=True)
        page_summary = gr.Group(visible=False)
        page_map     = gr.Group(visible=False)
        page_faults  = gr.Group(visible=False)
        page_modules = gr.Group(visible=False)

        # ---------------- DATA ----------------
        with page_data:
            gr.Markdown("### Data Loading", elem_classes=["section-title"])
            with gr.Row():
                uploads = gr.File(file_count="multiple", file_types=["image", ".json"], label="Images and GCPs.json")

            btn_copy = gr.Button("Load Files")
            log_copy = gr.Markdown()
            gcps_auto_info = gr.Markdown()
            det_json_path = gr.Textbox(visible=False)

            def _filter_and_copy(files):
                files = files or []
                def _name(x): return getattr(x, "name", str(x))
                exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif")
                imgs = [f for f in files if _name(f).lower().endswith(exts)]
                return validate_and_copy(imgs)

            def _load_files_and_gcps(files):
                run, log = _filter_and_copy(files)
                files = files or []
                def _name(x): return getattr(x, "name", str(x))
                jsons = [f for f in files if _name(f).lower().endswith(".json")]
                if jsons and run:
                    g_state, g_info = attach_gcps(run, jsons[0])
                else:
                    g_state, g_info = "", ""
                return run, log, g_state, g_info

            btn_copy.click(_load_files_and_gcps, inputs=[uploads],
                           outputs=[run_state, log_copy, gcps_state, gcps_auto_info])

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

                    simulate_ck.change(lambda v: gr.update(open=bool(v)), inputs=[simulate_ck], outputs=[sim_panel])
                    btn_init.click(sim_init, inputs=[run_state], outputs=[frame_dd, sim_img, sim_state, sim_help])

                    # IMPORTANT: use click_state (gr.State), not raw {}
                    sim_img.select(
                        sim_click,
                        inputs=[run_state, frame_dd, click_state, sim_state],
                        outputs=[sim_img, click_state, sim_state]
                    )
                    btn_undo.click(sim_undo, inputs=[run_state, frame_dd, sim_state], outputs=[sim_img, sim_state])
                    btn_clear.click(sim_clear, inputs=[run_state, frame_dd, sim_state], outputs=[sim_img, sim_state])
                    frame_dd.change(lambda run, name: sim_change_frame(run, name, sim_state.value or {})[0],
                                    inputs=[run_state, frame_dd], outputs=[sim_img])
                    btn_save.click(sim_save, inputs=[run_state, sim_state, sim_label, sim_score],
                                   outputs=[sim_save_info, det_json_path])

            with gr.Row(elem_classes=["nav-row"]):
                btn_nav_summary_from_data = gr.Button("View Summary", elem_classes=["navbtn"])
                gr.Button("View Panel Modules", elem_classes=["navbtn"]).click(
                    lambda: show_modules(),
                    outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )

        # ---------------- MAP ----------------
        with page_map:
            gr.Markdown("### Map", elem_classes=["section-title"])
            btn_stitch = gr.Button("Stitch & Project")
            pano_map = gr.Image(type="filepath", label="Panorama", interactive=True, height=640,
                                sources=[], show_download_button=True, show_fullscreen_button=True)
            map_info = gr.Markdown()
            with gr.Row():
                cb_open_maps = gr.Checkbox(value=False, label="Open Google Maps on click")
                cb_use_gcps  = gr.Checkbox(value=True,  label="Use GCPs")
                btn_save_img = gr.Button("Save Image")
                pano_file_dl = gr.File(label="Panorama (download)", interactive=False)

            btn_save_img.click(lambda p: p if p and os.path.isfile(p) else None,
                               inputs=[pano_png_state], outputs=[pano_file_dl])

            btn_stitch.click(lambda run: do_stitch_and_project(run, reuse=True),
                             inputs=[run_state],
                             outputs=[pano_map, map_info, pano_overlay_state, pano_png_state])

            pano_map.select(lambda run, open_maps, use_gcps: map_click(run, open_maps, 20, use_gcps),
                            inputs=[run_state, cb_open_maps, cb_use_gcps],
                            outputs=[map_info])

            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                btn_nav_summary_from_map = gr.Button("View Summary", elem_classes=["navbtn"])
                btn_nav_faults_from_map  = gr.Button("View Faults", elem_classes=["navbtn"])
                gr.Button("View Panel Modules", elem_classes=["navbtn"]).click(
                    lambda: show_modules(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )

        # ---------------- SUMMARY ----------------
        with page_summary:
            gr.Markdown("### Summary", elem_classes=["section-title"])
            btn_refresh   = gr.Button("Refresh")
            summary_md    = gr.Markdown()
            table         = gr.Dataframe(headers=["ImageIndex", "Class", "Severity", "Analysis"], row_count=10, wrap=True)
            btn_pdf       = gr.Button("Download Report (PDF)")
            pdf_file      = gr.File(label="Report", interactive=False)
            det_json_file = gr.File(label="Detections JSON", interactive=False)

            def _summary_and_det(run, tmp_path):
                md, rows = friendly_summary(run)
                if tmp_path and os.path.isfile(tmp_path):
                    det_path = tmp_path
                else:
                    det_path = os.path.join(run or "", "detections.json") if run else ""
                    if not (det_path and os.path.isfile(det_path)):
                        det_path = None
                return md, rows, det_path

            btn_refresh.click(_summary_and_det, inputs=[run_state, det_tmp_state], outputs=[summary_md, table, det_json_file])
            btn_pdf.click(lambda run: pdf_report(run), inputs=[run_state], outputs=[pdf_file])

            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                btn_nav_faults_from_summary = gr.Button("View Faults", elem_classes=["navbtn"])
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                gr.Button("View Panel Modules", elem_classes=["navbtn"]).click(
                    lambda: show_modules(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )

        # ---------------- FAULTS ----------------
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

            # FIX: outputs must be (cards, faults_state, faults_loaded_state)
            btn_load_faults.click(_load_faults, inputs=[run_state],
                                  outputs=[cards, faults_state, faults_loaded_state])

            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                btn_nav_summary_from_faults = gr.Button("View Summary", elem_classes=["navbtn"])
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                gr.Button("View Panel Modules", elem_classes=["navbtn"]).click(
                    lambda: show_modules(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )

        # ---------------- PANEL MODULES ----------------
        with page_modules:
            gr.Markdown("### Panel Modules (Thermography)", elem_classes=["section-title"])

            # Loader
            with gr.Row():
                modules_uploads = gr.File(file_count="multiple", file_types=["image"], label="Close-up images (IR or RGB)")
            with gr.Row():
                btn_modules_load = gr.Button("Load Close-up Files")
                modules_log = gr.Markdown()

            # Preprocessing
            with gr.Accordion("Preprocessing", open=False):
                with gr.Row():
                    pre_use_att = gr.Checkbox(value=False, label="use attention mask")
                    pre_scale   = gr.Slider(0.5, 3.0, value=1.75, step=0.05, label="scale")
                    pre_thresh  = gr.Slider(0, 255, value=140, step=1, label="binary / red threshold")
                with gr.Row():
                    pre_open    = gr.Number(value=1, label="morph opening (px)")
                    pre_close   = gr.Number(value=1, label="morph closing (px)")
                    pre_blur    = gr.Number(value=3, label="gaussian blur ksize (odd)")

            # Edges
            with gr.Accordion("Edges", open=False):
                with gr.Row():
                    edge_low   = gr.Number(value=30, label="Canny low (hysteresis_min_thresh)")
                    edge_high  = gr.Number(value=90, label="Canny high (hysteresis_max_thresh)")
                    edge_dil   = gr.Slider(0, 5, value=1, step=1, label="dilation steps")

            # Segments
            with gr.Accordion("Segments", open=True):
                with gr.Row():
                    delta_rho   = gr.Number(value=1,   label="delta rho")
                    delta_theta = gr.Number(value=1,   label="delta theta (deg)")
                    min_votes   = gr.Number(value=60,  label="min votes")
                    min_length  = gr.Number(value=50,  label="min length")
                    max_gap     = gr.Number(value=150, label="max gap")
                    extend_segs = gr.Number(value=10,  label="extend segments")
                with gr.Row():
                    cluster_type = gr.Radio(choices=["GMM","KNN"], value="GMM", label="type")
                    num_clusters = gr.Number(value=2, label="num clusters")
                    num_init     = gr.Number(value=5, label="num init (GMM)")
                    cluster_type.change(lambda t: gr.update(interactive=(t=="GMM")),
                                        inputs=[cluster_type], outputs=[num_init])
                with gr.Row():
                    feat_swipe   = gr.Checkbox(value=False, label="swipe")
                    feat_angles  = gr.Checkbox(value=True,  label="angles")
                    feat_centers = gr.Checkbox(value=False, label="centers")
                with gr.Row():
                    max_angle_var   = gr.Number(value=20, label="max angle variation")
                    max_merge_angle = gr.Number(value=10, label="max merging angle")
                    max_merge_dist  = gr.Number(value=10, label="max merging distance")

            # Actions
            with gr.Row():
                btn_modules_run = gr.Button("Run Thermography")
                btn_modules_zip = gr.Button("Save Results (ZIP)")
                modules_zip_out = gr.File(label="Results ZIP", interactive=False)

            # Outputs
            with gr.Row():
                modules_gallery = gr.Gallery(label="Detected Modules (Rectangles)", columns=[3], height=560)
                modules_json    = gr.JSON(label="Rectangles (per image)")

            with gr.Accordion("Diagnostics (first image)", open=False):
                with gr.Row():
                    diag_pre  = gr.Image(label="Preprocessed")
                    diag_att  = gr.Image(label="Attention Mask")
                with gr.Row():
                    diag_edge = gr.Image(label="Edges")
                    diag_segs = gr.Image(label="Segments Overlay")
                diag_log = gr.Markdown(label="Debug log")

            def _name(x): return getattr(x, "name", str(x))

            def _modules_copy(files):
                files = files or []
                exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif")
                imgs = [f for f in files if _name(f).lower().endswith(exts)]
                run, _log = validate_and_copy(imgs)

                paths = []
                if run and imgs:
                    for f in imgs:
                        bn = os.path.basename(_name(f))
                        p1 = os.path.join(run, "images", bn)
                        p2 = os.path.join(run, bn)
                        paths.append(p1 if os.path.isfile(p1) else p2)

                workdir = os.path.join(run or "", "_panel_modules")
                os.makedirs(workdir, exist_ok=True)

                msg = []
                if paths:
                    msg.append("Loaded close-up images:")
                    msg += [os.path.basename(p) for p in paths]
                else:
                    msg.append("No images found.")
                if not THERMO_OK:
                    msg += ["", "**Thermography repo not available**", THERMO_IMPORT_ERR]
                return run, "\n".join(msg), paths, workdir

            btn_modules_load.click(_modules_copy, inputs=[modules_uploads],
                                   outputs=[modules_run_state, modules_log, modules_files_state, modules_workdir])

            # Main run button → calls thermo_panel.run_modules(...)
            btn_modules_run.click(
                run_modules,
                inputs=[
                    modules_run_state, modules_files_state, modules_workdir,
                    # pre
                    pre_use_att, pre_scale, pre_thresh, pre_open, pre_close, pre_blur,
                    # edges
                    edge_low, edge_high, edge_dil,
                    # segments
                    delta_rho, delta_theta, min_votes, min_length, max_gap, extend_segs,
                    cluster_type, num_clusters, num_init, feat_swipe, feat_angles, feat_centers,
                    max_angle_var, max_merge_angle, max_merge_dist
                ],
                outputs=[modules_gallery, modules_json, modules_zip_out, diag_pre, diag_att, diag_edge, diag_segs, diag_log]
            )

            btn_modules_zip.click(
                lambda w: (os.path.join(w or "", "thermo_overlays.zip")
                           if w and os.path.isfile(os.path.join(w, "thermo_overlays.zip")) else None),
                inputs=[modules_workdir], outputs=[modules_zip_out]
            )

            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("Back to Data", elem_classes=["navbtn"]).click(
                    lambda: show_data(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                gr.Button("View Summary", elem_classes=["navbtn"]).click(
                    lambda: show_summary(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )
                gr.Button("View Faults", elem_classes=["navbtn"]).click(
                    lambda: show_faults(), outputs=[page_data, page_summary, page_map, page_faults, page_modules]
                )

        # -------- Summary nav (kept the same wiring as before)
        def _summary_and_det(run, tmp_path):
            md, rows = friendly_summary(run)
            if tmp_path and os.path.isfile(tmp_path):
                det_path = tmp_path
            else:
                det_path = os.path.join(run or "", "detections.json") if run else ""
                if not (det_path and os.path.isfile(det_path)):
                    det_path = None
            return md, rows, det_path

        def _nav_summary(run, tmp_path, sim, w, data, classes, imgsz, conf, iou, max_det):
            det_json = os.path.join(run or "", "detections.json") if run else ""
            if not (det_json and os.path.isfile(det_json)):
                w = w or "model.pt"
                try:
                    _msg, detp_new = run_detector_or_sim(run, bool(sim), w, data, classes, int(imgsz), float(conf), float(iou), int(max_det))
                    faults_loaded_state.value = False
                    det_json = detp_new or det_json
                    if detp_new and os.path.isfile(detp_new):
                        tdir = tempfile.mkdtemp(prefix="det_")
                        tpath = os.path.join(tdir, "detections.json")
                        shutil.copyfile(detp_new, tpath)
                        det_tmp_state.value = tpath
                except Exception:
                    pass

            v = (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible(False)), gr.update(visible=False))
            # small fix: gr.update requires keyword; correct tuple:
            v = (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True==False), gr.update(visible=False))  # placeholder; overwrite next line
            v = (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

            md, rows, detp = _summary_and_det(run, det_tmp_state.value or tmp_path)
            # First yield: show summary, don't change map yet
            yield (*v, md, rows, detp, gr.update(), gr.update())

            # Second yield: run stitch and push into map widgets so UI refreshes when done
            try:
                img_out, msg, ov_abs, pano_abs = do_stitch_and_project(run, reuse=True)
                pano_overlay_state.value = ov_abs
                pano_png_state.value     = pano_abs
                yield (*v, md, rows, detp, img_out, msg)
            except Exception:
                yield (*v, md, rows, detp, gr.update(), gr.update())

        def _nav_faults(run, loaded):
            v = (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))
            if loaded:
                # return gallery unchanged + states
                return (*v, gr.update(), faults_state.value, loaded)
            names, det_index, files = build_fault_index(run)
            gallery = build_fault_cards(run) if names else []
            return (*v, gallery, {"names": names, "index": det_index, "files": files}, bool(names))

        # wire buttons using the actual widgets as outputs (not placeholders)
        btn_nav_summary_from_data.click(
            _nav_summary,
            inputs=[run_state, det_tmp_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
            outputs=[page_data, page_summary, page_map, page_faults, page_modules, summary_md, table, det_json_file, pano_map, map_info]
        )
        btn_nav_summary_from_map.click(
            _nav_summary,
            inputs=[run_state, det_tmp_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
            outputs=[page_data, page_summary, page_map, page_faults, page_modules, summary_md, table, det_json_file, pano_map, map_info]
        )
        btn_nav_faults_from_map.click(
            _nav_faults,
            inputs=[run_state, faults_loaded_state],
            outputs=[page_data, page_summary, page_map, page_faults, page_modules, cards, faults_state, faults_loaded_state]
        )
        btn_nav_faults_from_summary.click(
            _nav_faults,
            inputs=[run_state, faults_loaded_state],
            outputs=[page_data, page_summary, page_map, page_faults, page_modules, cards, faults_state, faults_loaded_state]
        )
        btn_nav_summary_from_faults.click(
            _nav_summary,
            inputs=[run_state, det_tmp_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
            outputs=[page_data, page_summary, page_map, page_faults, page_modules, summary_md, table, det_json_file, pano_map, map_info]
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
