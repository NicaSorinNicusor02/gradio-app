# scripts/metadata_catalog.py
import os, glob
import pandas as pd
import exifread
from collections import defaultdict

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

def read_exif(path):
    out = dict(
        gps_lat=None, gps_lon=None,
        altitude_m=None,
        focal_length_mm=None,
        focal_35mm_equiv=None,
        sensor_w_mm=None, sensor_h_mm=None,
        heading_deg=None,
        notes=""
    )
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
        try:
            out['focal_length_mm'] = _rational_to_float(tags['EXIF FocalLength'])
        except Exception:
            pass

    if out['focal_length_mm'] is not None:
        if 4.3 <= float(out['focal_length_mm']) <= 4.5:
            out['focal_length_mm'] = 4.0

    if 'EXIF FocalLengthIn35mmFilm' in tags:
        try:
            out['focal_35mm_equiv'] = float(str(tags['EXIF FocalLengthIn35mmFilm']))
        except Exception:
            pass

    return out

def run_catalog(image_dir: str, output_csv: str, strict_extension: bool=True) -> pd.DataFrame:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"IMAGE_DIR nu există: {image_dir}")

    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) if strict_extension \
        else sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    groups = defaultdict(list)
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0].strip().lower()
        groups[stem].append(p)

    chosen = [sorted(paths)[0] for _, paths in groups.items()]

    rows = []
    for p in chosen:
        ex = read_exif(p)
        ex['image_name'] = os.path.basename(p)
        rows.append(ex)

    df = pd.DataFrame(rows, columns=[
        "image_name",
        "gps_lat", "gps_lon", "altitude_m",
        "focal_length_mm", "focal_35mm_equiv",
        "sensor_w_mm", "sensor_h_mm",
        "heading_deg",
        "notes"
    ])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df
