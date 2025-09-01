import os, cv2, numpy as np, contextlib

def imread_any(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0: return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def ensure_bgr(img):
    if img is None: return None
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def draw_segments_overlay(base_rgb, segments):
    vis = base_rgb.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR); vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    norm = []
    if segments is None:
        norm = []
    elif isinstance(segments, (list, tuple)):
        norm = list(segments)
    else:
        arr = np.asarray(segments)
        if arr.size == 0:
            norm = []
        elif arr.ndim == 2 and arr.shape[1] == 4:
            norm = [row for row in arr]
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            norm = [row.reshape(-1, 2) for row in arr]
        else:
            norm = [row for row in np.atleast_2d(arr)]
    for s in norm:
        try:
            a = np.array(s, dtype=float).reshape(-1, 2)
            if a.shape[0] >= 2:
                x1, y1 = a[0]; x2, y2 = a[1]
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        except Exception:
            continue
    return vis

def draw_rectangles_rgb(image_rgb, rectangles):
    vis = image_rgb.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR); vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    for rect in rectangles or []:
        pts = np.array(rect).astype(np.int32).reshape((-1,1,2))
        cv2.polylines(vis, [pts], True, (0,255,0), 2)
        for (x,y) in np.array(rect).astype(int):
            cv2.circle(vis, (x,y), 3, (255,0,0), -1)
    return vis

def img_stats(arr, name="img"):
    if arr is None: return f"{name}: None"
    a = np.asarray(arr)
    if a.size == 0: return f"{name}: empty"
    nz = int(np.count_nonzero(a))
    return f"{name}: shape={a.shape}, dtype={a.dtype}, min={a.min()}, max={a.max()}, mean={a.mean():.3f}, nonzero={nz} ({100.0*nz/max(1,a.size):.2f}%)"

def dump_params(obj, title="params"):
    if obj is None: return f"{title}: None"
    try: d = vars(obj)
    except Exception:
        d = {k:getattr(obj,k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj,k))}
    lines = [f"**{title}**"] + [f"- {k}: {d[k]}" for k in sorted(d.keys())]
    return "\n".join(lines)

@contextlib.contextmanager
def cv_findcontours_compat():
    orig = getattr(cv2, "findContours")
    def _fc(img, mode, method):
        res = orig(img, mode, method)
        if isinstance(res, tuple) and len(res) == 2:
            contours, hierarchy = res
            return (img, contours, hierarchy)
        return res
    try:
        cv2.findContours = _fc
        yield
    finally:
        cv2.findContours = orig
