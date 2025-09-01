import os, sys, json, zipfile
import numpy as np
import cv2
import gradio as gr

from thermo_bridge import repo, THERMO_OK, THERMO_IMPORT_ERR
from thermo_utils import (
    imread_any, ensure_bgr, draw_segments_overlay, draw_rectangles_rgb,
    img_stats, dump_params, cv_findcontours_compat
)

__all__ = ["THERMO_OK", "THERMO_IMPORT_ERR", "run_modules"]


def _apply_first_existing(obj, names, value):
    for n in names:
        if hasattr(obj, n):
            try:
                setattr(obj, n, value)
            except Exception:
                pass
            return True
    return False


def _run_one_debug(path, pre_kwargs, edge_kwargs, seg_kwargs):
    """Run full repo pipeline on one image with robust guards + debug artifacts."""
    R = repo()  # may raise if repo missing

    img = imread_any(path)
    if img is None:
        return None
    bgr = ensure_bgr(img)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    log_lines = [f"**File**: {os.path.basename(path)}", img_stats(rgb, "rgb")]

    with cv_findcontours_compat():
        # --- Preprocess
        p = R.PreprocessingParams()
        _apply_first_existing(p, ["use_attention", "enable_attention"], bool(pre_kwargs["use_att"]))
        _apply_first_existing(p, ["scale", "resize_factor", "image_scaling"], float(pre_kwargs["scale"]))
        _apply_first_existing(
            p, ["red_threshold", "bin_thresh", "threshold", "attention_threshold", "thresh"], int(pre_kwargs["thresh"])
        )
        _apply_first_existing(p, ["open_size", "opening_size", "morph_open", "open"], int(pre_kwargs["open"]))
        _apply_first_existing(p, ["close_size", "closing_size", "morph_close", "close"], int(pre_kwargs["close"]))
        _apply_first_existing(p, ["gaussian_blur", "blur_ksize", "gaussian_ksize", "blur"], int(pre_kwargs["blur"]))

        pre = R.FramePreprocessor(rgb, p)
        pre.preprocess()
        log_lines += [
            dump_params(p, "PreprocessingParams"),
            img_stats(pre.preprocessed_image, "preprocessed"),
            img_stats(pre.attention_image, "attention"),
        ]

        # --- Edges
        e = R.EdgeDetectorParams()
        _apply_first_existing(e, ["hysteresis_min_thresh", "low_threshold", "low", "canny_low"], int(edge_kwargs["low"]))
        _apply_first_existing(
            e, ["hysteresis_max_thresh", "high_threshold", "high", "canny_high"], int(edge_kwargs["high"])
        )
        _apply_first_existing(e, ["dilation_steps", "dilations", "dilate_steps"], int(edge_kwargs["dil"]))

        # Fallback to attention when mask is too sparse
        edge_input = pre.preprocessed_image
        try:
            nz_ratio = float((edge_input > 0).sum()) / max(1, edge_input.size)
        except Exception:
            nz_ratio = 0.0

        if nz_ratio < 0.20:  # tuneable threshold
            edge_input = pre.attention_image

        ed = R.EdgeDetector(edge_input, e)  # <-- fixed: use 'e' not 'eparams'
        ed.detect()
        log_lines += [dump_params(e, "EdgeDetectorParams"), img_stats(ed.edge_image, "edge")]

        # --- Segments
        s = R.SegmentDetectorParams()
        theta_rad = float(seg_kwargs["delta_theta"]) * np.pi / 180.0
        _apply_first_existing(s, ["delta_rho", "d_rho", "rho_resolution", "rho_step", "drho"], float(seg_kwargs["delta_rho"]))
        _apply_first_existing(s, ["delta_theta", "d_theta", "theta_resolution", "theta_step", "dtheta"], theta_rad)
        _apply_first_existing(s, ["min_votes", "min_num_votes", "threshold", "votes"], int(seg_kwargs["min_votes"]))
        _apply_first_existing(s, ["min_length", "min_line_length"], int(seg_kwargs["min_length"]))
        _apply_first_existing(s, ["max_gap", "max_line_gap"], int(seg_kwargs["max_gap"]))
        _apply_first_existing(s, ["extend_segments", "extension_pixels", "extend", "extend_len"], int(seg_kwargs["extend"]))

        sd = R.SegmentDetector(ed.edge_image, s)
        sd.detect()
        segs = getattr(sd, "segments", None)
        empty = (
            segs is None
            or (isinstance(segs, (list, tuple)) and len(segs) == 0)
            or (hasattr(segs, "size") and int(getattr(segs, "size")) == 0)
        )
        seg_count = 0 if empty else (len(segs) if isinstance(segs, (list, tuple)) else int(np.asarray(segs).shape[0]))
        log_lines += [dump_params(s, "SegmentDetectorParams"), f"segments: {seg_count}"]

        if empty:
            return {
                "rectangles": [],
                "preprocessed": pre.preprocessed_image,
                "attention": pre.attention_image,
                "edges": ed.edge_image,
                "segments_overlay": draw_segments_overlay(
                    pre.attention_image if pre.attention_image is not None else pre.preprocessed_image, []
                ),
                "debug": "\n\n".join(log_lines),
            }

        # --- Cluster
        c = R.SegmentClustererParams()
        algo = "gmm" if str(seg_kwargs["cluster_type"]).lower() == "gmm" else "knn"
        _apply_first_existing(c, ["type", "algo", "algorithm"], algo)
        _apply_first_existing(c, ["num_clusters", "n_clusters", "k", "clusters"], int(seg_kwargs["num_clusters"]))
        _apply_first_existing(c, ["num_init", "n_init"], int(seg_kwargs["num_init"]))
        _apply_first_existing(c, ["use_swipe", "swipe"], bool(seg_kwargs["feat_swipe"]))
        _apply_first_existing(c, ["use_angles", "angles_feature"], bool(seg_kwargs["feat_angles"]))
        _apply_first_existing(c, ["use_centers", "centers_feature"], bool(seg_kwargs["feat_centers"]))

        cl = R.SegmentClusterer(segs, c)
        try:
            cl.cluster_segments()
            mean_angles, _ = cl.compute_cluster_mean()
            cleanp = R.ClusterCleaningParams()
            _apply_first_existing(cleanp, ["max_angle_variation", "max_angle_var", "angle_var_max"], float(seg_kwargs["max_angle_var"]))
            _apply_first_existing(cleanp, ["max_merging_angle", "merge_angle_max"], float(seg_kwargs["max_merge_angle"]))
            _apply_first_existing(cleanp, ["max_merging_distance", "merge_dist_max"], float(seg_kwargs["max_merge_dist"]))
            cl.clean_clusters(mean_angles, cleanp)
            log_lines += [
                dump_params(c, "SegmentClustererParams"),
                dump_params(cleanp, "ClusterCleaningParams"),
                f"clusters: {len(getattr(cl, 'cluster_list', []) or [])}",
            ]
        except Exception as e:
            log_lines += [f"cluster_segments() failed: {e}"]

        # --- Intersections
        inter = R.IntersectionDetector(getattr(cl, "cluster_list", []), R.IntersectionDetectorParams())
        try:
            inter.detect()
            inter_cnt = len(getattr(inter, "cluster_cluster_intersections", []) or [])
            log_lines += [f"intersections: {inter_cnt}"]
        except Exception as e:
            log_lines += [f"intersections failed: {e}"]

        # --- Rectangles
        rd = R.RectangleDetector(getattr(inter, "cluster_cluster_intersections", []), R.RectangleDetectorParams())
        try:
            rd.detect()
            rects = getattr(rd, "rectangles", []) or []
            log_lines += [f"rectangles: {len(rects)}"]
        except Exception as e:
            rects = []
            log_lines += [f"rectangles detection failed: {e}"]

    return {
        "rectangles": rects,
        "preprocessed": pre.preprocessed_image,
        "attention": pre.attention_image,
        "edges": ed.edge_image,
        "segments_overlay": draw_segments_overlay(
            pre.attention_image if pre.attention_image is not None else pre.preprocessed_image, segs
        ),
        "debug": "\n\n".join(log_lines),
    }


def run_modules(
    run, paths, workdir,
    # Preproc
    use_att, scale, thresh, mopen, mclose, mblur,
    # Edges
    elow, ehigh, edil,
    # Segments
    drho, dtheta, mvotes, mlen, mgap, extend,
    ctype, nclus, ninit, fswipe, fangles, fcenters, maxavar, maxmang, maxmdist
):
    """
    Wired to the Gradio button in app.py.
    Returns: gallery, json, zip, diag_pre, diag_att, diag_edge, diag_segs, diag_log
    """
    if not THERMO_OK or not (run and paths):
        msg = ("Thermography repo not available: " + THERMO_IMPORT_ERR) if not THERMO_OK else "No run or files."
        return gr.update(), {}, None, None, None, None, None, msg

    pre_kwargs = dict(use_att=use_att, scale=scale, thresh=thresh, open=mopen, close=mclose, blur=mblur)
    edge_kwargs = dict(low=elow, high=ehigh, dil=edil)
    seg_kwargs = dict(
        delta_rho=drho,
        delta_theta=dtheta,
        min_votes=mvotes,
        min_length=mlen,
        max_gap=mgap,
        extend=extend,
        cluster_type=ctype,
        num_clusters=nclus,
        num_init=ninit,
        feat_swipe=fswipe,
        feat_angles=fangles,
        feat_centers=fcenters,
        max_angle_var=maxavar,
        max_merge_angle=maxmang,
        max_merge_dist=maxmdist,
    )

    os.makedirs(workdir, exist_ok=True)
    overlays = []
    results = {}
    first = None
    first_log = ""

    for pth in (paths or []):
        out = _run_one_debug(pth, pre_kwargs, edge_kwargs, seg_kwargs)
        if out is None:
            continue

        # Prefer attention for visualization (brighter), then preprocessed
        base = out["attention"] if out["attention"] is not None else out["preprocessed"]
        vis = base.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        for rect in out["rectangles"]:
            pts = np.array(rect).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            for (x, y) in np.array(rect).astype(int):
                cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)

        overlays.append(vis)
        results[os.path.basename(pth)] = [np.array(r).tolist() for r in (out["rectangles"] or [])]

        out_png = os.path.join(workdir, os.path.splitext(os.path.basename(pth))[0] + "_thermo.png")
        cv2.imwrite(out_png, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        if first is None:
            first = (out["preprocessed"], out["attention"], out["edges"], out["segments_overlay"])
            first_log = out.get("debug", "")

    with open(os.path.join(workdir, "thermography_rectangles.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    zip_path = os.path.join(workdir, "thermo_overlays.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in os.listdir(workdir):
            if name.endswith("_thermo.png"):
                zf.write(os.path.join(workdir, name), arcname=name)

    if first is None:
        return overlays, results, zip_path, None, None, None, None, "No outputs."

    pre_i, att_i, edg_i, seg_i = first
    return overlays, results, zip_path, pre_i, att_i, edg_i, seg_i, first_log
