import os
import webbrowser
import gradio as gr
import pandas as pd
from PIL import Image

# --- căi implicite (schimbă după nevoie) ---
DATA_DIR = os.path.join("data")
INPUT_DIR = os.path.join(DATA_DIR, "input_images")
OUT_DIR   = os.path.join(DATA_DIR, "outputs")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --- import wrappers ---
from scripts_app.metadata_catalog import run_catalog
from scripts_app.yolo_geolocatie import run_yolo_to_csv
from scripts_app.stitch import build_panorama
from scripts_app.viz_defecte import draw_pins

# =========================
# TAB 1 — Catalog EXIF
# =========================
def ui_run_catalog(image_dir, out_csv, strict_ext):
    df = run_catalog(image_dir, out_csv, strict_extension=strict_ext)
    return out_csv, df

# =========================
# TAB 2 — YOLO→GPS→CSV
# =========================
def ui_run_yolo(
    image_dir, model_path, metadata_csv, out_csv,
    conf_thres, iou_thres, ref_lat, ref_lon,
    origin_image_name, agl_override_m, focal_override_mm,
    match_by_stem, ignore_exif
):
    df = run_yolo_to_csv(
        image_dir=image_dir,
        model_path=model_path,
        metadata_csv=metadata_csv,
        output_csv=out_csv,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        ref_bl_lat=ref_lat,
        ref_bl_lon=ref_lon,
        origin_image_name=(origin_image_name or None),
        agl_override_m=(agl_override_m if agl_override_m is not None else None),
        focal_override_mm=(focal_override_mm if focal_override_mm is not None else None),
        match_by_stem=bool(match_by_stem),
        ignore_exif=bool(ignore_exif),
    )
    return out_csv, df


# =========================
# TAB 3 — Panoramă
# =========================
def ui_build_pano(image_dir, sp_w, sg_w, proj_root, pano_out, target_h, target_w):
    path = build_panorama(
        image_dir=image_dir,
        superpoint_weights=sp_w,
        superglue_weights=sg_w,
        project_root=proj_root,
        output_path=pano_out,
        target_h=int(target_h), target_w=int(target_w),
    )
    return path

# =========================
# TAB 4 — Vizualizare
# =========================
def ui_draw_pins(pano_path, csv_path):
    img, df, has_pano = draw_pins(pano_path, csv_path)
    return img, df, ("Da" if has_pano else "Nu")

def ui_click_on_image(evt: gr.SelectData, df_csv):
    if df_csv is None or len(df_csv)==0:
        return "Nu am rânduri în CSV."
    # dacă există coordonate pe panoramă, alegi pinul cel mai apropiat; altfel doar raportezi click
    if "pano_x_px" in df_csv.columns and "pano_y_px" in df_csv.columns:
        cx, cy = evt.index
        pins = df_csv.dropna(subset=["pano_x_px","pano_y_px"]).copy()
        if pins.empty:
            return "Nu există pini în CSV."
        d2 = (pins["pano_x_px"] - cx)**2 + (pins["pano_y_px"] - cy)**2
        idx = int(d2.idxmin())
        row = df_csv.loc[idx]
    else:
        # fără coordonate, alegi primul (sau nimic)
        row = df_csv.iloc[0]

    lat = float(row.get("defect_lat"))
    lon = float(row.get("defect_lon"))
    cls = row.get("defect_class","?")
    conf = row.get("confidence","?")
    url = f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"
    try: webbrowser.open_new_tab(url)
    except: pass
    return f"#{int(getattr(row,'name',0))} | {cls} ({conf:.2f}) → [Deschide în Google Maps]({url})"

# =========================
# UI Gradio
# =========================
with gr.Blocks(title="Drone CV Suite", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🚁 Drone CV Suite — Catalog EXIF · YOLO→GPS · Panoramă · Vizualizare")

    state_df = gr.State(None)    # pentru tabelul încărcat în Tab 4

    with gr.Tabs():
        # ---------- TAB 1 ----------
        with gr.Tab("1) Catalog EXIF → CSV"):
            with gr.Row():
                t_image_dir = gr.Textbox(value=INPUT_DIR, label="Director imagini")
                t_out_csv   = gr.Textbox(value=os.path.join(OUT_DIR, "catalog_metadate.csv"), label="CSV ieșire")
            strict_ext = gr.Checkbox(value=True, label="Doar .jpg lowercase (STRICT_EXTENSION)")
            btn_cat = gr.Button("Generează catalog")
            cat_file = gr.File(label="CSV generat")
            cat_table = gr.Dataframe(label="Previzualizare CSV", wrap=True)

            btn_cat.click(ui_run_catalog, [t_image_dir, t_out_csv, strict_ext], [cat_file, cat_table])

        # ---------- TAB 2 ----------
        with gr.Tab("2) Detectare YOLO → GPS → CSV"):
            with gr.Row():
                y_img_dir = gr.Textbox(value=INPUT_DIR, label="Director imagini")
                y_model   = gr.Textbox(value="best_model.pt", label="Model YOLO (.pt)")
            with gr.Row():
                y_meta    = gr.Textbox(value=os.path.join(OUT_DIR, "catalog_metadate.csv"), label="CSV metadate (din Tab 1)")
                y_out     = gr.Textbox(value=os.path.join(OUT_DIR, "rezultate_defecte.csv"), label="CSV rezultate")
            with gr.Row():
                y_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.01, label="CONF_THRES")
                y_iou  = gr.Slider(0.05, 0.9, value=0.45, step=0.01, label="IOU_THRES")
            match_by_stem = gr.Checkbox(
                value=True,
                label="Potrivește metadatele din catalog după 'stem' (nume fără extensie)"
            )
            ignore_exif = gr.Checkbox(
                value=True,
                label="Ignoră EXIF (folosește doar catalogul)"
            )
    
            gr.Markdown("**Referință BL (colț jos‑stânga a primei imagini)**")
            with gr.Row():
                ref_lat = gr.Number(value=44.880175, label="REF_BL_LAT")
                ref_lon = gr.Number(value=25.533855, label="REF_BL_LON")
            with gr.Accordion("Opțional", open=False):
                origin_name = gr.Textbox(value="", label="ORIGIN_IMAGE_NAME (exact)")
                agl_override = gr.Number(value=None, label="AGL_OVERRIDE_M (m) — dacă vrei să înlocuiești altitudinea")
                focal_override = gr.Number(value=4.0, label="FOCAL_OVERRIDE_MM (mm) sau None")

            btn_yolo = gr.Button("Rulează YOLO→GPS")
            yolo_file = gr.File(label="CSV rezultat")
            yolo_table = gr.Dataframe(label="Previzualizare rezultate", wrap=True)

            btn_yolo.click(
                ui_run_yolo,
                [y_img_dir, y_model, y_meta, y_out, y_conf, y_iou, ref_lat, ref_lon, origin_name, agl_override, focal_override, match_by_stem, ignore_exif],
                [yolo_file, yolo_table]
            ).then(lambda df: df, inputs=[yolo_table], outputs=[state_df])

        # ---------- TAB 3 ----------
        with gr.Tab("3) Construiește panoramă (SuperGlue)"):
            with gr.Row():
                p_img_dir = gr.Textbox(value=INPUT_DIR, label="Director imagini panoramă")
                sp_w = gr.Textbox(value="", label="SUPERPOINT_WEIGHTS")
            with gr.Row():
                sg_w = gr.Textbox(value="", label="SUPERGLUE_WEIGHTS")
                proj_root = gr.Textbox(value="", label="PROJECT_ROOT (repo SuperGlue)")
            with gr.Row():
                p_out = gr.Textbox(value=os.path.join(OUT_DIR, "panorama.png"), label="Ieșire panoramă")
            with gr.Row():
                th = gr.Slider(600, 2400, value=1200, step=10, label="TARGET_H")
                tw = gr.Slider(800, 4000, value=1600, step=10, label="TARGET_W")

            btn_pano = gr.Button("Generează panoramă")
            pano_preview = gr.Image(label="Previzualizare panoramă", interactive=False)

            btn_pano.click(ui_build_pano, [p_img_dir, sp_w, sg_w, proj_root, p_out, th, tw], [pano_preview])

        # ---------- TAB 4 ----------
        with gr.Tab("4) Vizualizare • panoramă + defecte"):
            with gr.Row():
                v_pano = gr.Image(type="filepath", label="Panoramă (PNG/JPG)")
                v_csv  = gr.File(label="CSV defecte (din Tab 2)")

            btn_draw = gr.Button("Încarcă și desenează pini")
            pano_with_pins = gr.Image(label="Panoramă cu pini (click pentru Maps)", interactive=True)
            has_pano_flag = gr.Textbox(label="CSV conține coloane pano_x_px / pano_y_px?", interactive=False)
            csv_view = gr.Dataframe(label="CSV", wrap=True)
            click_info = gr.Markdown()

            def _load_and_draw(pano_path, csv_file):
                path = csv_file.name if csv_file else ""
                img, df, has_pano = draw_pins(pano_path, path)
                return img, ("Da" if has_pano else "Nu"), df, df

            btn_draw.click(_load_and_draw, [v_pano, v_csv], [pano_with_pins, has_pano_flag, csv_view, state_df])

            pano_with_pins.select(ui_click_on_image, [state_df], [click_info])

    gr.Markdown("—\nTips: setează căile absolute Windows (ex: `C:\\\\Users\\\\Sorin\\\\...`).")

if __name__ == "__main__":
    demo.launch()
