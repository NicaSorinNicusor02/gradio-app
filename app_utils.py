import os, json, shutil, time, uuid
from typing import List, Tuple, Optional
import cv2 as cv

# --- config & lifecycle ---
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "app_runs")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# recreate runs dir on import
shutil.rmtree(RUNS_DIR, ignore_errors=True)
os.makedirs(RUNS_DIR, exist_ok=True)


def new_run_dir() -> str:
    d = os.path.join(RUNS_DIR, time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "det_vis"), exist_ok=True)
    return d

# --- path helpers ---
def overlay_path(pano_png: str) -> str: return os.path.splitext(pano_png)[0] + ".overlay.png"
def boxes_path(pano_png: str)   -> str: return os.path.splitext(pano_png)[0] + ".boxes.json"
def world_path(pano_png: str)   -> str: return os.path.splitext(pano_png)[0] + ".world.json"
def index_path(pano_png: str)   -> str: return os.path.splitext(pano_png)[0] + ".index.json"
def gcps_path(run_dir: str)     -> str: return os.path.join(run_dir, "gcps.json")

# --- I/O helpers ---
def gcps_count(path: Optional[str]) -> int:
    try:
        if not path: return 0
        g = json.load(open(path, "r"))
        return sum(1 for r in g if all(k in r for k in ("u", "v", "lat", "lon")))
    except Exception:
        return 0


def sorted_images(imgs_dir: str) -> List[str]:
    if not os.path.isdir(imgs_dir): return []
    allf = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir)
            if f.lower().endswith(IMG_EXTS)]
    return sorted(allf, key=lambda p: os.path.basename(p).lower())


def validate_and_copy(files: List) -> Tuple[str, str]:
    if not files:
        return "", "No files uploaded."
    run = new_run_dir()
    imgs_dir = os.path.join(run, "images")
    ok, bad = 0, []
    for f in files:
        name = os.path.basename(getattr(f, "name", "img.jpg"))
        if not name.lower().endswith(IMG_EXTS):
            bad.append(f"{name}: unsupported extension")
            continue
        dst = os.path.join(imgs_dir, name)
        try:
            shutil.copy(f.name, dst)
            im = cv.imread(dst, cv.IMREAD_COLOR)
            if im is None or im.size == 0:
                bad.append(f"{name}: corrupted")
                os.remove(dst)
            else:
                ok += 1
        except Exception as e:
            bad.append(f"{name}: copy/read error ({e})")
    msg = [f"Copied {ok} images.", f"Run: `{run}`"]
    if bad:
        msg.append("Issues:\n" + "\n".join([f"- {x}" for x in bad]))
    return run, "\n".join(msg)


def attach_gcps(run: str, gcps_file) -> Tuple[str, str]:
    if not run: return "", "Upload images first."
    if not gcps_file: return "", "No GCP file provided."
    try:
        dst = gcps_path(run)
        shutil.copy(gcps_file.name, dst)
        return dst, f"GCPs saved ({gcps_count(dst)} points)."
    except Exception as e:
        return "", f"Failed to save GCPs: {e}"
