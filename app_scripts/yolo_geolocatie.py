# scripts/yolodetect_to_gps.py
import os, glob
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
from ultralytics import YOLO
import exifread
import cv2

# ====== UTILITARE din scriptul tău, neschimbate (copiate) ======
def _rational_to_float(x):
    try:
        return float(x.num) / float(x.den)
    except Exception:
        try:
            n, d = str(x).split('/')
            return float(n) / float(d)
        except Exception:
            return float(x)

def _dms_to_deg(dms, ref):
    deg = _rational_to_float(dms[0]) + _rational_to_float(dms[1]) / 60.0 + _rational_to_float(dms[2]) / 3600.0
    return -deg if ref in ['S', 'W'] else deg

def read_exif(path: str) -> Dict[str, Optional[float]]:
    out = dict(gps_lat=None, gps_lon=None, altitude_m=None, focal_length_mm=None, focal_35mm_equiv=None)
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return out
    glat, glatr = tags.get('GPS GPSLatitude'), tags.get('GPS GPSLatitudeRef')
    glon, glonr = tags.get('GPS GPSLongitude'), tags.get('GPS GPSLongitudeRef')
    if glat and glatr and glon and glonr:
        try:
            out['gps_lat'] = _dms_to_deg(glat.values, glatr.values)
            out['gps_lon'] = _dms_to_deg(glon.values, glonr.values)
        except Exception:
            pass
    galt, galtref = tags.get('GPS GPSAltitude'), tags.get('GPS GPSAltitudeRef')
    if galt:
        try:
            alt = _rational_to_float(getattr(galt, 'values', [galt])[0])
            ref = 0
            if galtref:
                try: ref = int(str(getattr(galtref, 'values', [0])[0]))
                except Exception: ref = 0
            out['altitude_m'] = -alt if ref == 1 else alt
        except Exception:
            pass
    if out['altitude_m'] is None:
        try:
            for k in tags.keys():
                if 'RelativeAltitude' in k:
                    try:
                        out['altitude_m'] = float(str(tags[k])); break
                    except Exception:
                        pass
        except Exception:
            pass
    if 'EXIF FocalLength' in tags:
        try: out['focal_length_mm'] = _rational_to_float(tags['EXIF FocalLength'])
        except Exception: pass
    if 'EXIF FocalLengthIn35mmFilm' in tags:
        try: out['focal_35mm_equiv'] = float(str(tags['EXIF FocalLengthIn35mmFilm']))
        except Exception: pass
    return out

def estimate_sensor_from_35mm(focal_mm: float, focal_35mm: float) -> Tuple[Optional[float], Optional[float]]:
    if focal_mm and focal_35mm and focal_35mm > 0:
        crop = focal_35mm / focal_mm
        return 36.0 / crop, 24.0 / crop
    return None, None

def compute_gsd(alt_m: float, focal_mm: float, sensor_w_mm: float, sensor_h_mm: float,
                img_w_px: int, img_h_px: int) -> Tuple[Optional[float], Optional[float]]:
    if None in [alt_m, focal_mm, sensor_w_mm, sensor_h_mm]:
        return None, None
    f_m  = focal_mm / 1000.0
    sw_m = sensor_w_mm / 1000.0
    sh_m = sensor_h_mm / 1000.0
    return (alt_m * sw_m) / (f_m * img_w_px), (alt_m * sh_m) / (f_m * img_h_px)

def latlon_to_enu_m(lat, lon, lat0, lon0):
    latm = np.deg2rad((lat + lat0) / 2.0)
    east  = (lon - lon0) * (111320.0 * np.cos(latm))
    north = (lat - lat0) * 110574.0
    return east, north

def enu_m_to_latlon(east_m, north_m, lat0, lon0):
    dlat = north_m / 110574.0
    dlon = east_m  / (111320.0 * np.cos(np.deg2rad(lat0)))
    return lat0 + dlat, lon0 + dlon

def imgvec_to_enu(x_right_m: float, y_down_m: float, heading_deg: float):
    psi = np.deg2rad(heading_deg)
    east  =  x_right_m * np.cos(psi) - y_down_m * np.sin(psi)
    north = -x_right_m * np.sin(psi) - y_down_m * np.cos(psi)
    return east, north

def normalize_angle_deg(a):
    a = a % 360.0
    if a < 0: a += 360.0
    return a

def bearing_deg_between(lat1, lon1, lat2, lon2):
    de, dn = latlon_to_enu_m(lat2, lon2, lat1, lon1)
    ang = np.degrees(np.arctan2(de, dn))
    return normalize_angle_deg(ang)

def load_catalog(path: str, index_by_stem: bool = False):
    if not path or not os.path.isfile(path):
        print(f"[INFO] METADATA_CSV absent sau cale greșită:", path)
        return {}
    df = pd.read_csv(path)
    if 'image_name' not in df.columns:
        raise RuntimeError("CSV-ul trebuie să aibă coloana 'image_name'.")
    df['image_name'] = df['image_name'].astype(str).str.strip()

    # cheie: exact image_name sau stem (fără extensie)
    def make_key(name: str) -> str:
        name = str(name).strip()
        if index_by_stem:
            import os
            return os.path.splitext(os.path.basename(name))[0].lower()
        else:
            return name.lower()

    catalog = {}
    for _, r in df.iterrows():
        key = make_key(r['image_name'])
        catalog[key] = {
            'gps_lat': r.get('gps_lat', np.nan),
            'gps_lon': r.get('gps_lon', np.nan),
            'altitude_m': r.get('altitude_m', np.nan),
            'focal_length_mm': r.get('focal_length_mm', np.nan),
            'focal_35mm_equiv': r.get('focal_35mm_equiv', np.nan),
            'sensor_w_mm': r.get('sensor_w_mm', np.nan),
            'sensor_h_mm': r.get('sensor_h_mm', np.nan),
        }
    print(f"[OK] Catalog încărcat (cheie={'stem' if index_by_stem else 'image_name'}):", path, f"(rânduri: {len(catalog)})")
    return catalog


def merge_meta(img_name: str, exif: dict, cat: dict,
               agl_override_m: Optional[float],
               focal_override_mm: Optional[float],
               default_focal=4.0, default_sw=6.0, default_sh=4.0,
               match_by_stem: bool = False,
               ignore_exif: bool = False):
    import os
    name = str(img_name).strip()
    key = os.path.splitext(os.path.basename(name))[0].lower() if match_by_stem else name.lower()
    row = cat.get(key, {}) if cat else {}

    # helper: ia CSV dacă există; altfel EXIF (dacă nu e ignorat); altfel default
    def pick(keyname, default=None):
        v_csv  = row.get(keyname, np.nan)
        v_exif = None if ignore_exif else (exif.get(keyname) if exif else None)
        if (not pd.isna(v_csv)) if isinstance(v_csv, (float, int)) else (v_csv is not np.nan):
            return v_csv
        return v_exif if v_exif is not None else default

    alt_used = agl_override_m if (agl_override_m is not None) else pick('altitude_m')

    meta = {
        'gps_lat':          pick('gps_lat'),
        'gps_lon':          pick('gps_lon'),
        'altitude_m':       alt_used,
        'focal_length_mm':  pick('focal_length_mm', default_focal),
        'focal_35mm_equiv': pick('focal_35mm_equiv'),
        'sensor_w_mm':      pick('sensor_w_mm', default_sw),
        'sensor_h_mm':      pick('sensor_h_mm', default_sh),
    }

    if focal_override_mm is not None:
        meta['focal_length_mm'] = float(focal_override_mm)

    if (meta['sensor_w_mm'] is None or meta['sensor_h_mm'] is None) and meta['focal_length_mm'] and meta['focal_35mm_equiv']:
        sw, sh = estimate_sensor_from_35mm(meta['focal_length_mm'], meta['focal_35mm_equiv'])
        meta['sensor_w_mm'] = meta['sensor_w_mm'] or sw
        meta['sensor_h_mm'] = meta['sensor_h_mm'] or sh

    return meta


def run_inference(model, img_path: str, conf_thres: float, iou_thres: float):
    results = model.predict(source=img_path, conf=conf_thres, iou=iou_thres, verbose=False)
    dets = []
    names = model.names
    im = cv2.imread(img_path)
    h, w = im.shape[:2]
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            cls_idx = int(b.cls[0].cpu().numpy())
            dets.append([x1, y1, x2, y2, conf, cls_idx])
    dets = np.array(dets) if dets else np.zeros((0,6))
    return dets, names, (w, h)

# ====== WRAPPER principal apelabil din Gradio ======
def run_yolo_to_csv(
    image_dir: str,
    model_path: str,
    metadata_csv: str,
    output_csv: str,
    conf_thres: float=0.25,
    iou_thres: float=0.45,
    ref_bl_lat: float=44.880175,
    ref_bl_lon: float=25.533855,
    origin_image_name: Optional[str]=None,
    agl_override_m: Optional[float]=None,
    focal_override_mm: Optional[float]=4.0,
    match_by_stem: bool = True,
    ignore_exif: bool = True,
) -> pd.DataFrame:

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        raise RuntimeError(f"Nu s-au găsit imagini .jpg în {image_dir}")

    catalog = load_catalog(metadata_csv, index_by_stem=match_by_stem) if metadata_csv and os.path.isfile(metadata_csv) else {}

    model = YOLO(model_path)

    metas = []
    for p in images:
        name = os.path.basename(p)
        exif_i = read_exif(p)
        meta_i = merge_meta(name, exif_i, catalog,
                    agl_override_m, focal_override_mm,
                    match_by_stem=match_by_stem,
                    ignore_exif=ignore_exif)

        metas.append((p, name, meta_i))

    # origine — prima cu GPS+GSD valid sau numele dat
    origin_idx = None
    if origin_image_name:
        normalized = origin_image_name.strip().lower()
        for i, (_, name, _) in enumerate(metas):
            if name.strip().lower() == normalized:
                origin_idx = i; break
        if origin_idx is None:
            raise RuntimeError(f"ORIGIN_IMAGE_NAME='{origin_image_name}' nu a fost găsit în folder.")
    if origin_idx is None:
        for i, (p, name, m) in enumerate(metas):
            if m.get('gps_lat') is None or m.get('gps_lon') is None:
                continue
            im = cv2.imread(p);  h, w = im.shape[:2]
            gsdx_i, gsdy_i = compute_gsd(m.get('altitude_m'), m.get('focal_length_mm'),
                                         m.get('sensor_w_mm'), m.get('sensor_h_mm'), w, h)
            if gsdx_i is not None and gsdy_i is not None:
                origin_idx = i; break
    if origin_idx is None:
        raise RuntimeError("Nicio imagine nu are GPS + GSD valid. Completează CSV (lat/lon, altitudine, focală, senzor).")

    first_path, first_name, origin_meta = metas[origin_idx]
    im0 = cv2.imread(first_path);  h0, w0 = im0.shape[:2]

    alt0 = origin_meta['altitude_m']
    gsdx0_est, gsdy0_est = compute_gsd(alt0, origin_meta['focal_length_mm'],
                                       origin_meta['sensor_w_mm'], origin_meta['sensor_h_mm'], w0, h0)
    if gsdx0_est is None or gsdy0_est is None:
        raise RuntimeError("Nu pot calcula GSD pentru origine. Verifică altitudine/focală/senzor.")
    gsd0_iso_est = float(np.sqrt(abs(gsdx0_est * gsdy0_est)))

    lat_cam0, lon_cam0 = origin_meta['gps_lat'], origin_meta['gps_lon']
    dE_known, dN_known = latlon_to_enu_m(ref_bl_lat, ref_bl_lon, lat_cam0, lon_cam0)

    vx, vy = -w0/2.0, +h0/2.0
    A = dE_known * vx - dN_known * vy
    B = -(dE_known * vy + dN_known * vx)
    psi_cal = normalize_angle_deg(np.degrees(np.arctan2(B, A)))

    c = np.cos(np.deg2rad(psi_cal)); s = np.sin(np.deg2rad(psi_cal))
    bE = vx * c - vy * s
    bN = -vx * s - vy * c
    s_cal = (dE_known * bE + dN_known * bN) / (vx*vx + vy*vy)

    k_scale = float(s_cal / gsd0_iso_est)

    bl0_e_cal = s_cal * bE
    bl0_n_cal = s_cal * bN

    def prev_valid(i):
        for k in range(i-1, -1, -1):
            if metas[k][2].get('gps_lat') is not None and metas[k][2].get('gps_lon') is not None:
                return k
        return None
    def next_valid(i):
        for k in range(i+1, len(metas)):
            if metas[k][2].get('gps_lat') is not None and metas[k][2].get('gps_lon') is not None:
                return k
        return None

    bearings = [None]*len(metas)
    for i in range(len(metas)):
        lat_i, lon_i = metas[i][2].get('gps_lat'), metas[i][2].get('gps_lon')
        p, n = prev_valid(i), next_valid(i)
        if p is not None and n is not None:
            bearings[i] = bearing_deg_between(metas[p][2]['gps_lat'], metas[p][2]['gps_lon'],
                                              metas[n][2]['gps_lat'], metas[n][2]['gps_lon'])
        elif p is not None:
            bearings[i] = bearing_deg_between(metas[p][2]['gps_lat'], metas[p][2]['gps_lon'], lat_i, lon_i)
        elif n is not None:
            bearings[i] = bearing_deg_between(lat_i, lon_i, metas[n][2]['gps_lat'], metas[n][2]['gps_lon'])
        else:
            bearings[i] = psi_cal

    bearing_origin = bearings[origin_idx] if bearings[origin_idx] is not None else psi_cal
    yaw_offset = normalize_angle_deg(psi_cal - bearing_origin)

    rows: List[Dict] = []
    for idx, (img_path, name, meta) in enumerate(metas, 1):
        det, names, (img_w, img_h) = run_inference(model, img_path, conf_thres, iou_thres)
        bearing_i = bearings[idx-1]
        heading_i = normalize_angle_deg(bearing_i + yaw_offset) if bearing_i is not None else psi_cal

        gsdx, gsdy = compute_gsd(meta.get('altitude_m'), meta.get('focal_length_mm'),
                                 meta.get('sensor_w_mm'), meta.get('sensor_h_mm'), img_w, img_h)
        if gsdx is not None and gsdy is not None:
            gsdx *= k_scale; gsdy *= k_scale
        else:
            gsdx = gsdy = s_cal

        lat_i, lon_i = meta.get('gps_lat'), meta.get('gps_lon')
        if lat_i is None or lon_i is None:
            continue
        cam_e, cam_n = latlon_to_enu_m(lat_i, lon_i, lat_cam0, lon_cam0)

        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls_idx = det[i]
            cls_idx = int(cls_idx)
            cls_name = names.get(cls_idx, str(cls_idx))
            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0

            dx_img_m = (cx - (img_w / 2.0)) * gsdx
            dy_img_m = (cy - (img_h / 2.0)) * gsdy
            de_e, de_n = imgvec_to_enu(dx_img_m, dy_img_m, heading_i)

            defect_e_global = cam_e + de_e
            defect_n_global = cam_n + de_n

            x_from_ref_bl = defect_e_global - bl0_e_cal
            y_from_ref_bl = defect_n_global - bl0_n_cal

            defect_lat, defect_lon = enu_m_to_latlon(defect_e_global, defect_n_global, lat_cam0, lon_cam0)

            rows.append({
                "image_name": name,
                "defect_class": cls_name,
                "confidence": float(conf),
                "bbox_xmin_px": float(x1), "bbox_ymin_px": float(y1),
                "bbox_xmax_px": float(x2), "bbox_ymax_px": float(y2),
                "center_x_px": float(cx), "center_y_px": float(cy),
                "img_width_px": float(img_w), "img_height_px": float(img_h),

                "gps_lat_camera": lat_i, "gps_lon_camera": lon_i,
                "altitude_m": meta.get('altitude_m'),
                "focal_length_mm": meta.get('focal_length_mm'),
                "focal_35mm_equiv": meta.get('focal_35mm_equiv'),
                "sensor_w_mm": meta.get('sensor_w_mm'), "sensor_h_mm": meta.get('sensor_h_mm'),
                "heading_deg_used": heading_i,

                "x_m_from_ref_bl": x_from_ref_bl,
                "y_m_from_ref_bl": y_from_ref_bl,

                "defect_lat": defect_lat,
                "defect_lon": defect_lon,
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    df.to_csv(output_csv, index=False, encoding='utf-8')
    return df
