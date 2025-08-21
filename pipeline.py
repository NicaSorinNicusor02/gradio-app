#!/usr/bin/env python3
import os, json, argparse, cv2 as cv
import numpy as np
from pathlib import Path as PPath
import sys
from core.stitcher import Stitcher
from core.mapper   import GeoMapper
from core.detector import YOLODetector
from core.simulator import DetectionSimulator
from core.box_projector import PanoBoxProjector
from core.geolocalizer import PanoGeoLocalizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")

def _normalize_out_path(path_like, default_name="pano.png"):
    p = PPath(path_like)
    exts = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}
    if p.is_dir(): p = p / default_name
    elif p.suffix.lower() not in exts: p = p.with_suffix(".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

def save_index_json(out_path, pano_size_wh, H_proc_to_pano, frames):
    idx = {
        "mosaic_size": [int(pano_size_wh[0]), int(pano_size_wh[1])],
        "H_proc_to_pano": [np.array(H, dtype=np.float64).tolist() for H in H_proc_to_pano],
        "frames": frames
    }
    with open(out_path, "w") as f:
        json.dump(idx, f, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Full pipeline: Step1 stitch + Step2 detect/simulate + project + Step3 geolocalize")
    ap.add_argument("images_dir", help="Folder with input images (e.g., ./images)")
    ap.add_argument("--out", default="out/pano.png", help="Output pano image path OR directory")

    # Step1 options
    ap.add_argument("--no_seams", action="store_true")
    ap.add_argument("--exposure", action="store_true")
    ap.add_argument("--num_bands", type=int, default=3)
    ap.add_argument("--orb_fallback", action="store_true")

    # Step2 options
    ap.add_argument("--step2", choices=["none","detect","simulate"], default="none")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--device", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_vis", default=None)
    ap.add_argument("--sim_label", default="manual")
    ap.add_argument("--sim_score", type=float, default=1.0)
    ap.add_argument("--sim_start", type=int, default=0)
    ap.add_argument("--sim_end", type=int, default=None)

    # Step3 options
    ap.add_argument("--step3", choices=["none","geo"], default="none",
                    help="Open interactive geolocalizer (click to get lat/lon; click boxes to open their center)")
    ap.add_argument("--boxes_path", default=None, help="If not given, uses <pano>.boxes.json")
    ap.add_argument("--gcps", default=None, help="Optional GCPs JSON to refine pixel→world")
    ap.add_argument("--openmaps", action="store_true", help="Open Google Maps on click")
    ap.add_argument("--zoom", type=int, default=20, help="Maps zoom level")
    ap.add_argument("--uv", default=None, help="Query a specific pixel (u,v) without UI, e.g. --uv 1234,567")

    args = ap.parse_args()

    # Normalize out path and derive all sidecars
    args.out = _normalize_out_path(args.out)
    base = os.path.splitext(args.out)[0]
    overlay_path = base + ".overlay.png"
    world_path   = base + ".world.json"
    index_path   = base + ".index.json"
    boxes_path   = args.boxes_path or (base + ".boxes.json")

    # ---------------- Step 1: stitch + yaw-only + world + index ----------------
    stitcher = Stitcher(
        imgs_dir=args.images_dir,
        max_kpts=2000,
        yaw_align=True,
        exposure=args.exposure,
        no_seams=args.no_seams,
        num_bands=args.num_bands,
        orb_fallback=args.orb_fallback
    )
    pano, meta, exifs = stitcher.run()

    gmap = GeoMapper(align='yaw')
    phi_deg, A_world, epsg = gmap.choose_rotation(meta, exifs)
    pano_oriented, A_adj, epsg_out, world_json_path, M_pano, _ = gmap.orient_and_world(
        pano, phi_deg, A_world, epsg, args.out
    )

    ok = cv.imwrite(args.out, pano_oriented)
    if not ok:
        raise RuntimeError(f"Failed to write pano to '{args.out}'. Use a supported extension like .png or .jpg.")

    Hs = [np.array(H, dtype=np.float64) for H in meta["Hs_shift"]]
    if M_pano is None: M_pano = np.eye(3, dtype=np.float64)
    Hs_out = [(np.array(M_pano, dtype=np.float64) @ H).tolist() for H in Hs]
    save_index_json(index_path, pano_oriented.shape[1::-1], Hs_out, meta.get("frame_models", []))

    print(f"[OK] Step1 panorama: {args.out}")
    if os.path.exists(world_path): print(f"[OK] world: {world_path}")
    print(f"[OK] index: {index_path}")

    # ---------------- Step 2: detect/simulate + project ----------------
    if args.step2 != "none":
        idx = json.load(open(index_path, "r"))

        if args.step2 == "detect":
            if not args.weights:
                raise SystemExit("--weights is required for --step2 detect")
            det = YOLODetector(weights=args.weights, device=args.device, imgsz=args.imgsz,
                               conf=args.conf, iou=args.iou, max_det=args.max_det,
                               classes=args.classes, data_yaml=args.data)
            detections = det.run_on_index(idx, limit=args.limit, save_vis_dir=args.save_vis)
        else:
            sim = DetectionSimulator()
            detections = sim.run(idx, out_path=base + ".detections.json",
                                 start=args.sim_start, end=args.sim_end,
                                 label=args.sim_label, score=args.sim_score)

        pj = PanoBoxProjector(idx, pano_bgr=cv.imread(args.out, cv.IMREAD_COLOR))
        boxes_on_pano = pj.project(detections)
        pj.write_overlay(overlay_path)
        with open(boxes_path, "w") as f: json.dump(boxes_on_pano, f, indent=2)
        print(f"[OK] Step2 overlay: {overlay_path}")
        print(f"[OK] Step2 boxes:   {boxes_path}")

    # ---------------- Step 3: interactive geolocalizer ----------------
    if args.step3 == "geo":
        # Programmatic query of a pixel without UI
        if args.uv:
            u, v = map(float, args.uv.split(","))
            gl = PanoGeoLocalizer(args.out, world_json=world_path, gcps_json=args.gcps,
                                  boxes_json=boxes_path, open_on_click=args.openmaps, zoom=args.zoom)
            lat, lon, E, N = gl.uv_to_latlon(u, v)
            print(f"[Step3] u={u}, v={v} -> lat={lat:.7f}, lon={lon:.7f}  (E={E:.2f}, N={N:.2f})")
            if args.openmaps: import webbrowser; webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={lat:.7f}%2C{lon:.7f}&zoom={args.zoom}")
        else:
            gl = PanoGeoLocalizer(args.out, world_json=world_path, gcps_json=args.gcps,
                                  boxes_json=boxes_path, open_on_click=args.openmaps, zoom=args.zoom)
            gl.run()

if __name__ == "__main__":
    main()