# scripts/viewer_utils.py
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

REQUIRED_COLS = ["defect_lat", "defect_lon", "defect_class", "confidence", "image_name"]
PANO_COLS = ["pano_x_px", "pano_y_px"]

def draw_pins(pano_path: str, csv_path: str, pin_radius: int = 8):
    if not os.path.exists(pano_path):
        raise FileNotFoundError(f"Panorama inexistentă: {pano_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV inexistent: {csv_path}")

    pano = Image.open(pano_path).convert("RGB")
    draw = ImageDraw.Draw(pano)
    df = pd.read_csv(csv_path)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            # nu blocăm complet – afișăm totuși CSV-ul
            pass

    has_pano = all(c in df.columns for c in PANO_COLS)
    if has_pano:
        w, h = pano.size
        for _, r in df.dropna(subset=PANO_COLS).iterrows():
            x = int(np.clip(r["pano_x_px"], 0, w-1))
            y = int(np.clip(r["pano_y_px"], 0, h-1))
            bbox = (x-pin_radius, y-pin_radius, x+pin_radius, y+pin_radius)
            draw.ellipse(bbox, outline=(255, 0, 0), width=3)

    return pano, df, has_pano
