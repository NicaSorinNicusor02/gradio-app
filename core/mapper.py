# core/mapper.py
import json, os
import numpy as np
import cv2 as cv
from pyproj import CRS, Transformer
from .geometry import estimate_geo, angle_from_yaw, rotate_affine  # ← no flips imported

class GeoMapper:
    def __init__(self, align='yaw'):
        self.align = align

    def choose_rotation(self, meta, exifs):
        """
        Pick the visual rotation angle (yaw/gps/auto), but ALWAYS try to compute
        (A_world, epsg) so we can emit pano.world.json even when aligning by yaw.
        """
        A = None; epsg = None; phi_gps = None; phi_yaw = None

        # Always attempt GPS-based world affine (does not affect the chosen angle)
        try:
            A, phi_gps, _, epsg = estimate_geo(meta['Hs_shift'], exifs)
        except Exception:
            A, phi_gps, epsg = None, None, None

        # Yaw heading for visual alignment
        try:
            phi_yaw = angle_from_yaw(exifs)
        except Exception:
            phi_yaw = None

        # Choose angle per mode, but carry (A, epsg) regardless
        if self.align == 'gps' and phi_gps is not None: return phi_gps, A, epsg
        if self.align == 'yaw' and phi_yaw is not None: return phi_yaw, A, epsg
        if phi_gps is not None and phi_yaw is not None:
            d = ((phi_gps - phi_yaw + 180) % 360) - 180
            return (phi_yaw if abs(d) > 45 else phi_gps), A, epsg
        if phi_yaw is not None: return phi_yaw, A, epsg
        if phi_gps is not None: return phi_gps, A, epsg
        return 0.0, A, epsg

    def orient_and_world(self, pano, phi_deg, A_world, epsg, out_path):

        pano_rot, M = rotate_affine(pano, -phi_deg)
        R3 = np.eye(3); R3[:2, :] = M

        jf = None; M_pano = R3  # no flip => final pano transform is just the rotation

        # If we have a world affine, express it in the rotated pano frame and write world.json
        if A_world is not None and epsg is not None:
            A3 = np.eye(3); A3[:2, :] = A_world
            A_world = (A3 @ np.linalg.inv(R3))[:2, :]
            meta = {
                "A_2x3": A_world.tolist(),
                "crs_epsg": int(epsg),
                "axes": {"pixel_x": "right/East", "pixel_y": "down", "north_up_visual": True}
            }
            jf = out_path.rsplit('.', 1)[0] + ".world.json"
            with open(jf, "w") as f:
                json.dump(meta, f, indent=2)

        # Return: (oriented pano, world affine in oriented pano frame, epsg, path, M_pano, flip=False)
        return pano_rot, A_world, epsg, jf, M_pano, False


class ClickMapper:
    def __init__(self, world_json, gcps_json=None):
        m = json.load(open(world_json, "r"))
        self.A = np.array(m["A_2x3"], dtype=np.float64)
        self.epsg = int(m["crs_epsg"])
        self.tf_EN_WGS = Transformer.from_crs(CRS.from_epsg(self.epsg), CRS.from_epsg(4326), always_xy=True)
        self.tf_WGS_EN = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(self.epsg), always_xy=True)
        self.H = self._homography_from_gcps(gcps_json)

    def _homography_from_gcps(self, gcps_json):
        if not gcps_json or not os.path.exists(gcps_json): return None
        g = json.load(open(gcps_json, "r"))
        src, dst = [], []
        for r in g:
            if not all(k in r for k in ("u","v","lat","lon")): continue
            E, N = self.tf_WGS_EN.transform(r["lon"], r["lat"])
            src.append([float(r["u"]), float(r["v"])])
            dst.append([float(E), float(N)])
        if len(src) < 4: return None
        H, _ = cv.findHomography(np.asarray(src,np.float64), np.asarray(dst,np.float64), method=0)
        return H

    def uv_to_EN(self, u, v):
        if self.H is not None:
            p = self.H @ np.array([u, v, 1.0], np.float64)
            E = p[0] / p[2]; N = p[1] / p[2]
        else:
            A = self.A
            E = A[0,0]*u + A[0,1]*v + A[0,2]
            N = A[1,0]*u + A[1,1]*v + A[1,2]
        return float(E), float(N)

    def uv_to_latlon(self, u, v):
        E, N = self.uv_to_EN(u, v)
        lon, lat = self.tf_EN_WGS.transform(E, N)
        return float(lat), float(lon), float(E), float(N)
