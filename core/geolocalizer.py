#!/usr/bin/env python3
# core/geolocalizer.py
import os, json, webbrowser
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2 as cv
from .mapper import ClickMapper

def _draw_label(img, text, org, color, scale=0.95, thick=3):
    (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1]); pad = 6
    cv.rectangle(img, (x-2, y-th-pad), (x+tw+2, y+base), (0,0,0), -1, cv.LINE_AA)
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)

def _load_boxes(path: Optional[str]) -> List[Dict]:
    if path and os.path.exists(path):
        return json.load(open(path, "r"))
    return []

def _point_in_quad(quad: np.ndarray, x: int, y: int) -> bool:
    # quad: (4,2)
    return cv.pointPolygonTest(quad.astype(np.int32), (int(x), int(y)), False) >= 0

def _open_maps(lat: float, lon: float, zoom: int):
    webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={lat:.7f}%2C{lon:.7f}&zoom={zoom}")

class PanoGeoLocalizer:
    """
    Interactive geolocalizer for the stitched panorama.

    - Uses ClickMapper(world_json, gcps_json) to convert pano pixels to (lat, lon).
    - If boxes_json is provided (from Step 2 projection), clicking a box opens
      its center on Maps (optional) and prints coordinates.
    - Clicking anywhere returns exact lat/lon for that pixel.
    """
    def __init__(self,
                 pano_path: str,
                 world_json: Optional[str] = None,
                 gcps_json: Optional[str] = None,
                 boxes_json: Optional[str] = None,
                 open_on_click: bool = False,
                 zoom: int = 20) -> None:
        self.pano_path = pano_path
        self.world_json = world_json or pano_path.rsplit('.', 1)[0] + ".world.json"
        self.gcps_json = gcps_json
        self.boxes_json = boxes_json
        self.open_on_click = open_on_click
        self.zoom = int(zoom)

        if not os.path.exists(self.world_json):
            raise SystemExit(f"[geo] world file not found: {self.world_json}")

        self.cm = ClickMapper(self.world_json, self.gcps_json)
        self.pano = cv.imread(self.pano_path, cv.IMREAD_COLOR)
        if self.pano is None:
            raise SystemExit(f"[geo] cannot read pano image: {self.pano_path}")

        self.H, self.W = self.pano.shape[:2]
        self.boxes = _load_boxes(self.boxes_json)
        # normalize ids
        for k, b in enumerate(self.boxes):
            if "id" not in b: b["id"] = k

    # ---------- programmatic APIs ----------
    def uv_to_latlon(self, u: float, v: float) -> Tuple[float, float, float, float]:
        """Returns (lat, lon, E, N) for pano pixel (u,v)."""
        return self.cm.uv_to_latlon(u, v)

    def box_center_latlon(self, box_id: int) -> Tuple[float, float, float, float]:
        """Returns (lat, lon, E, N) for the center of a given box id (from boxes_json)."""
        for b in self.boxes:
            if int(b.get("id", -1)) == int(box_id) and "quad" in b:
                q = np.asarray(b["quad"], np.float64)
                cx, cy = float(q[:,0].mean()), float(q[:,1].mean())
                return self.cm.uv_to_latlon(cx, cy)
        raise ValueError(f"box id {box_id} not found or has no 'quad'.")

    # ---------- interactive UI ----------
    def run(self) -> None:
        winW, winH = min(1600, self.W), min(1000, self.H)
        cx, cy, z = self.W/2.0, self.H/2.0, 1.0
        state = {"last": None, "sel": None}
        base = self._draw_boxes(self.pano, self.boxes)

        def clamp(v, a, b): return max(a, min(b, v))
        def vp():
            vw = max(100, int(self.W / z)); vh = max(100, int(self.H / z))
            x0 = int(clamp(cx - vw/2, 0, self.W - vw))
            y0 = int(clamp(cy - vh/2, 0, self.H - vh))
            return x0, y0, vw, vh
        def s2i(px, py, x0, y0, vw, vh):
            sx = vw / float(winW); sy = vh / float(winH)
            return float(x0 + px * sx), float(y0 + py * sy)
        def render():
            x0, y0, vw, vh = vp()
            crop = base[y0:y0+vh, x0:x0+vw]
            disp = cv.resize(crop, (winW, winH), interpolation=cv.INTER_LINEAR)
            if state["last"] is not None:
                u, v, lat, lon, _, _ = state["last"]
                px = int((u - x0) * winW / float(vw)); py = int((v - y0) * winH / float(vh))
                cv.circle(disp, (px, py), 10, (0,255,255), 4, cv.LINE_AA)
                _draw_label(disp, f"{lat:.7f}, {lon:.7f}", (px + 15, py - 15), (0,255,255), 0.9, 3)
            return disp

        def on_click(evt, x, y, flags, param):
            if evt != cv.EVENT_LBUTTONDOWN: return
            x0, y0, vw, vh = vp()
            u, v = s2i(x, y, x0, y0, vw, vh)
            hit = self._hit_box(int(round(u)), int(round(v)))
            if hit is not None:
                q = np.asarray(hit["quad"], np.float64)
                cxq, cyq = float(q[:,0].mean()), float(q[:,1].mean())
                lat, lon, E, N = self.cm.uv_to_latlon(cxq, cyq)
                state["last"] = (cxq, cyq, lat, lon, E, N); state["sel"] = hit.get("id")
                print(f"[geo] box #{hit.get('id')} -> lat={lat:.7f}, lon={lon:.7f}")
                if self.open_on_click: _open_maps(lat, lon, self.zoom)
            else:
                lat, lon, E, N = self.cm.uv_to_latlon(u, v)
                state["last"] = (u, v, lat, lon, E, N); state["sel"] = None
                print(f"[geo] lat={lat:.7f}, lon={lon:.7f}")
                if self.open_on_click: _open_maps(lat, lon, self.zoom)

        win = "Step3: click on box (center) or anywhere. +/- zoom, WASD pan, R reset, Q quit"
        cv.namedWindow(win, cv.WINDOW_NORMAL); cv.resizeWindow(win, winW, winH)
        cv.setMouseCallback(win, on_click)

        while True:
            disp = render(); cv.imshow(win, disp)
            k = cv.waitKey(30) & 0xFF
            if k in (ord('q'), 27): break
            if k in (ord('+'), ord('=')): z = min(16.0, z * 1.25)
            if k in (ord('-'), ord('_')): z = max(1.0, z / 1.25)
            step = max(20, int(min(self.W, self.H) / (10 * z)))
            if k in (ord('a'), ord('A')): cx -= step
            if k in (ord('d'), ord('D')): cx += step
            if k in (ord('w'), ord('W')): cy -= step
            if k in (ord('s'), ord('S')): cy += step
            if k in (ord('r'), ord('R')): cx, cy, z = self.W/2.0, self.H/2.0, 1.0
            cx = clamp(cx, 0, self.W); cy = clamp(cy, 0, self.H)
        cv.destroyAllWindows()

    # ---------- helpers ----------
    def _draw_boxes(self, img, boxes, sel=None):
        out = img.copy()
        for b in boxes:
            if "quad" not in b: continue
            pts = np.array(b["quad"], np.int32).reshape(-1,1,2)
            color = (0,255,0) if b.get("id") != sel else (0,0,255)
            cv.polylines(out, [pts], True, color, 4, cv.LINE_AA)
            cx, cy = pts[:,0,0].mean(), pts[:,0,1].mean()
            _draw_label(out, f'#{b.get("id","")}', (int(cx), int(cy)), color, 1.0, 3)
        return out

    def _hit_box(self, x: int, y: int) -> Optional[Dict]:
        for b in self.boxes:
            if "quad" not in b: continue
            q = np.asarray(b["quad"], np.int32)
            if _point_in_quad(q, x, y):
                return b
        return None
