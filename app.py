#!/usr/bin/env python3
import os, sys
import gradio as gr

# Make sure core/ is importable when running from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from utils import validate_and_copy, attach_gcps
from services import (
    run_detector_or_sim, sim_init, sim_change_frame, sim_click, sim_undo, sim_clear, sim_save,
    friendly_summary, pdf_report, do_stitch_and_project, map_click,
    faults_table, build_fault_index, render_fault_image,
)

# ---- Old color template (dark) ----
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

/* Bottom nav buttons — make them pop a bit */
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
"""

def build_ui():
    with gr.Blocks(title="Solar Inspection", theme=theme, css=CSS) as demo:
        gr.HTML('<div class="header"><h1>Solar Inspection</h1></div>')

        # ---------- Reactive state ----------
        run_state          = gr.State("")
        gcps_state         = gr.State("")
        click_state        = gr.State({})
        sim_state          = gr.State({})
        faults_state       = gr.State({"names": [], "index": {}, "files": []})
        current_fault_idx  = gr.State(0)
        pano_overlay_state = gr.State("")
        pano_png_state     = gr.State("")

        # ---------- Simple nav helpers (toggle page visibility) ----------
        def show_data():    return (gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        def show_summary(): return (gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False))
        def show_map():     return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),  gr.update(visible=False))
        def show_faults():  return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

        # ---------- Declare pages first ----------
        page_data    = gr.Group(visible=True)
        page_summary = gr.Group(visible=False)
        page_map     = gr.Group(visible=False)
        page_faults  = gr.Group(visible=False)

        # =================== PAGE: DATA ===================
        with page_data:
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

            btn_copy.click(lambda files: validate_and_copy(files), inputs=[uploads], outputs=[run_state, log_copy])
            btn_gcps.click(attach_gcps, inputs=[run_state, gcps_file], outputs=[gcps_state, gcps_info])

            def _run_det_or_sim(run, sim, w, data, classes, imgsz, conf, iou, max_det):
                msg, detp = run_detector_or_sim(run, sim, w, data, classes, int(imgsz), float(conf), float(iou), int(max_det))
                return msg, detp
            btn_detect.click(
                _run_det_or_sim,
                inputs=[run_state, simulate_ck, y_weights, y_data, y_classes, y_imgsz, y_conf, y_iou, y_maxd],
                outputs=[det_status, det_json_path]
            )

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
                gr.Button("View Summary", elem_classes=["navbtn"]).click(
                    lambda: show_summary(), outputs=[page_data, page_summary, page_map, page_faults]
                )

        # =================== PAGE: MAP (reverted with zoom slider) ===================
        with page_map:
            gr.Markdown("### Map", elem_classes=["section-title"])
            btn_stitch = gr.Button("Stitch & Project")

            # Use a plain Image that expects a *filepath*
            pano_map   = gr.Image(type="filepath", label="Panorama", interactive=True, height=640)
            map_info   = gr.Markdown()
            with gr.Row():
                cb_open_maps = gr.Checkbox(value=False, label="Open Google Maps on click")
                cb_use_gcps  = gr.Checkbox(value=True,  label="Use GCPs")
                zoom_slider  = gr.Slider(10, 22, value=20, step=1, label="Zoom")

            # IMPORTANT: wire outputs to the Image, not an HTML component
            btn_stitch.click(
                lambda run: do_stitch_and_project(run, reuse=True),
                inputs=[run_state],
                outputs=[pano_map, map_info, pano_overlay_state, pano_png_state],  # Image, Markdown, State, State
            )

            # Clicking on the panorama -> lat/lon + optional browser open
            pano_map.select(
                map_click,
                inputs=[run_state, cb_open_maps, zoom_slider, cb_use_gcps],
                outputs=[map_info],
            )

            # Bottom nav buttons
            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("View Summary", elem_classes=["navbtn"]).click(
                    lambda: show_summary(), outputs=[page_data, page_summary, page_map, page_faults]
                )
                gr.Button("View Faults", elem_classes=["navbtn"]).click(
                    lambda: show_faults(), outputs=[page_data, page_summary, page_map, page_faults]
                )

        # =================== PAGE: SUMMARY ===================
        with page_summary:
            gr.Markdown("### Summary", elem_classes=["section-title"])
            btn_refresh = gr.Button("Refresh")
            summary_md = gr.Markdown()
            table = gr.Dataframe(headers=["ImageIndex", "Class", "Severity", "Analysis"], row_count=10, wrap=True)
            btn_pdf   = gr.Button("Download Report (PDF)")
            pdf_file  = gr.File(label="Report", interactive=False)

            btn_refresh.click(lambda run: friendly_summary(run), inputs=[run_state], outputs=[summary_md, table])
            btn_pdf.click(lambda run: pdf_report(run), inputs=[run_state], outputs=[pdf_file])

            # Bottom nav buttons
            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults]
                )
                gr.Button("View Faults", elem_classes=["navbtn"]).click(
                    lambda: show_faults(), outputs=[page_data, page_summary, page_map, page_faults]
                )

        # =================== PAGE: FAULTS ===================
        with page_faults:
            gr.Markdown("### Faults", elem_classes=["section-title"])
            btn_load_faults = gr.Button("Load")
            faults_md = gr.Markdown()
            faults_tbl = gr.Dataframe(headers=["Image", "Class", "Score", "Severity", "Analysis"], row_count=10, wrap=True)
            with gr.Row():
                img_name_dd = gr.Dropdown(choices=[], label="Image")
                btn_prev_img = gr.Button("Prev"); btn_next_img = gr.Button("Next")
            img_preview = gr.Image(label="", interactive=False, height=560)

            def _load_faults(run):
                names, det_index, files = build_fault_index(run)
                if names:
                    rows_all, msg, _ = faults_table(run)
                    first_idx_val = names[0][1]
                    img, _rows_first = render_fault_image(files, first_idx_val, det_index)
                    return (msg, rows_all,
                            gr.update(choices=[n for n, _ in names], value=names[0][0]),
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
                map_names = {n: i for n, i in names}
                idx = map_names.get(name, names[0][1])
                img, rows = render_fault_image(files, idx, det_index)
                return img, idx

            img_name_dd.change(_select_name, inputs=[img_name_dd, faults_state],
                               outputs=[img_preview, current_fault_idx])

            def _step(delta, state, cur_idx):
                names = state.get("names", [])
                files = state.get("files", [])
                det_index = state.get("index", {})
                if not names: return gr.update(), cur_idx, None
                order = [i for _, i in names]
                if cur_idx not in order: cur_idx = order[0]
                j = max(0, min(len(order) - 1, order.index(cur_idx) + delta))
                new_idx = order[j]
                new_name = [n for n, i in names if i == new_idx][0]
                img, rows = render_fault_image(files, new_idx, det_index)
                return gr.update(value=new_name), new_idx, img

            btn_prev_img.click(lambda *args: _step(-1, *args),
                               inputs=[faults_state, current_fault_idx],
                               outputs=[img_name_dd, current_fault_idx, img_preview])
            btn_next_img.click(lambda *args: _step(+1, *args),
                               inputs=[faults_state, current_fault_idx],
                               outputs=[img_name_dd, current_fault_idx, img_preview])

            # Bottom nav buttons
            with gr.Row(elem_classes=["nav-row"]):
                gr.Button("View Summary", elem_classes=["navbtn"]).click(
                    lambda: show_summary(), outputs=[page_data, page_summary, page_map, page_faults]
                )
                gr.Button("View Map", elem_classes=["navbtn"]).click(
                    lambda: show_map(), outputs=[page_data, page_summary, page_map, page_faults]
                )

    return demo

if __name__ == "__main__":
    build_ui().launch()
