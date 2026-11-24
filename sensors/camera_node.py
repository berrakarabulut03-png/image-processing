"""
RealSense D455F (Color + Depth) + OpenCV buoy detection (improved)

İyileştirmeler:
- wait_for_frames timeout yükseltildi (default 400ms)
- başlangıç warmup (ilk frame gelmeme sorununu azaltır)
- depth=0 gelince 5x5 komşulukta medyan ile robust mesafe
- daha sağlam morfoloji + min area filtresi
- opsiyonel preview overlay (enable_preview=True)

Gerekenler:
pip install pyrealsense2 opencv-python numpy
"""

from typing import Dict, Tuple, List
import math
import time

try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2
    _HAS_RS = True
except Exception:
    rs = None
    np = None
    cv2 = None
    _HAS_RS = False


class CameraNode:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_preview: bool = False,
        timeout_ms: int = 400,
        warmup_frames: int = 30,
        min_area: float = 250.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_preview = enable_preview
        self.timeout_ms = timeout_ms
        self.warmup_frames = warmup_frames
        self.min_area = min_area

        self.pipeline = None
        self.align = None
        self.intr = None  # color intrinsics
        self.last_frame = None
        self.last_overlay = None

        # HSV aralıkları (duba renklerine göre ayarlayabilirsin)
        # Kırmızı iki aralık (0-10 ve 160-180)
        self.hsv_ranges = {
            "red": [
                ((0, 120, 70), (10, 255, 255)),
                ((160, 120, 70), (180, 255, 255)),
            ],
            "orange": [
                ((10, 120, 70), (22, 255, 255)),
            ],
            "yellow": [
                ((22, 120, 70), (38, 255, 255)),
            ],
            "green": [
                ((40, 80, 70), (85, 255, 255)),
            ],
        }

        if not _HAS_RS:
            print("[CAM] pyrealsense2/opencv/numpy yok -> Dummy moda geçiliyor.")
            return

        try:
            print("[CAM] RealSense pipeline başlatılıyor...")
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            profile = self.pipeline.start(config)

            # Depth'i color'a hizala
            self.align = rs.align(rs.stream.color)

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intr = color_stream.get_intrinsics()

            print(
                "[CAM] RealSense başladı. Intrinsics:",
                f"fx={self.intr.fx:.2f}, fy={self.intr.fy:.2f}, ppx={self.intr.ppx:.2f}, ppy={self.intr.ppy:.2f}",
            )

            # Warmup: ilk frameler bazen geç gelir / boş gelir
            for _ in range(self.warmup_frames):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
                except Exception:
                    pass

        except Exception as e:
            print(f"[CAM] RealSense başlatılamadı: {e} -> Dummy moda geçiliyor.")
            self.pipeline = None

    # ---------- helpers ----------

    def _make_mask(self, hsv_img, ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]):
        mask = None
        for lo, hi in ranges:
            lo_np = np.array(lo, dtype=np.uint8)
            hi_np = np.array(hi, dtype=np.uint8)
            m = cv2.inRange(hsv_img, lo_np, hi_np)
            mask = m if mask is None else (mask | m)
        return mask

    def _detect_buoy(self, color_bgr) -> Tuple[bool, float, int, int, str]:
        """
        Döner: (found, area, cx, cy, color_name)
        En büyük alanlı (en güvenilir) aday seçilir.
        """
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        best = (False, 0.0, 0, 0, "")

        kernel = np.ones((5, 5), np.uint8)

        for name, ranges in self.hsv_ranges.items():
            mask = self._make_mask(hsv, ranges)

            # temizleme
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            if area > best[1]:
                best = (True, area, cx, cy, name)

        return best

    def _pixel_angle_deg(self, cx: int) -> float:
        """
        Intrinsics ile doğru yatay açı:
        angle = atan((cx - ppx) / fx)
        """
        if self.intr is None:
            return 0.0
        x = (cx - self.intr.ppx) / self.intr.fx
        return math.degrees(math.atan(x))

    def _robust_depth(self, depth_frame, cx: int, cy: int) -> float:
        """
        depth 0/boş gelebilir.
        5x5 komşulukta (sıfır olmayan) değerlerin medyanını al.
        """
        vals = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x = max(0, min(self.width - 1, cx + dx))
                y = max(0, min(self.height - 1, cy + dy))
                d = float(depth_frame.get_distance(x, y))
                if d > 0.05:
                    vals.append(d)

        if not vals:
            return 0.0
        vals.sort()
        return vals[len(vals) // 2]

    # ---------- main API ----------

    def process_frame(self) -> Dict[str, float]:
        """
        Çıkış:
        {
          buoy_detected: bool,
          buoy_color: str,
          buoy_angle_deg: float,
          buoy_distance_m: float,
          buoy_area: float
        }
        """
        # Dummy (placeholder gibi rastgele true üretmesin diye hep false)
        if self.pipeline is None:
            return {
                "buoy_detected": False,
                "buoy_color": "",
                "buoy_angle_deg": 0.0,
                "buoy_distance_m": 0.0,
                "buoy_area": 0.0,
            }

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return {
                    "buoy_detected": False,
                    "buoy_color": "",
                    "buoy_angle_deg": 0.0,
                    "buoy_distance_m": 0.0,
                    "buoy_area": 0.0,
                }

            color_img = np.asanyarray(color_frame.get_data())  # BGR
            self.last_frame = color_img.copy()
            self.last_overlay = color_img.copy()

            found, area, cx, cy, cname = self._detect_buoy(color_img)

            if not found:
                if self.enable_preview:
                    cv2.imshow("Color", color_img)
                    cv2.waitKey(1)
                return {
                    "buoy_detected": False,
                    "buoy_color": "",
                    "buoy_angle_deg": 0.0,
                    "buoy_distance_m": 0.0,
                    "buoy_area": 0.0,
                }

            angle_deg = self._pixel_angle_deg(cx)
            dist_m = float(self._robust_depth(depth_frame, cx, cy))

            if self.enable_preview:
                vis = color_img.copy()
                cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    f"{cname} a={angle_deg:.1f}deg d={dist_m:.2f}m area={area:.0f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                self.last_overlay = vis.copy()
                cv2.imshow("Color", vis)
                cv2.waitKey(1)

            return {
                "buoy_detected": True,
                "buoy_color": cname,
                "buoy_angle_deg": float(angle_deg),
                "buoy_distance_m": float(dist_m),
                "buoy_area": float(area),
            }

        except Exception as e:
            # İlk anda frame gecikmesi vs. görülebilir; spam azaltmak için istersen yorumlayabilirsin.
            print(f"[CAM] Frame hata: {e}")
            time.sleep(0.02)
            return {
                "buoy_detected": False,
                "buoy_color": "",
                "buoy_angle_deg": 0.0,
                "buoy_distance_m": 0.0,
                "buoy_area": 0.0,
            }

    
    def process_detections(self) -> List[Dict[str, float]]:
        """
        Çoklu duba çıktısı (parkur gate/engel için).
        Çıkış listesi:
          [{"color": str, "angle_deg": float, "distance_m": float, "area": float}, ...]
        Not: pipeline yoksa [] döner.
        """
        if self.pipeline is None:
            return []

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return []

            color = np.asanyarray(color_frame.get_data())
            self.last_frame = color.copy()
            self.last_overlay = color.copy()
            depth = np.asanyarray(depth_frame.get_data())

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            kernel = np.ones((5, 5), np.uint8)

            dets: List[Dict[str, float]] = []

            for name, ranges in self.hsv_ranges.items():
                mask = self._make_mask(hsv, ranges)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    area = float(cv2.contourArea(c))
                    if area < self.min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)

                    ang = float(self._pixel_angle_deg(cx))
                    dist = float(self._robust_depth(depth_frame, cx, cy))

                    # depth bazen 0 gelebilir; 0 ise mesafe bilinmiyor (0 bırakma)
                    if dist <= 0.05:
                        dist = 0.0

                    dets.append({
                        "color": str(name),
                        "angle_deg": ang,
                        "distance_m": dist,
                        "area": area,
                    })

            # sort by area desc
            dets.sort(key=lambda d: float(d.get("area", 0.0)), reverse=True)

            # optional preview
            if self.enable_preview:
                overlay = color.copy()
                for d in dets[:8]:
                    txt = f'{d["color"]} {d["angle_deg"]:+.1f}deg {d["distance_m"]:.1f}m'
                    cv2.putText(overlay, txt, (10, 30 + 20 * dets.index(d)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                self.last_overlay = overlay.copy()
                cv2.imshow("Buoy detections", overlay)
                cv2.waitKey(1)

            return dets

        except Exception:
            return []

    def get_last_overlay_frame(self):
        return self.last_overlay if self.last_overlay is not None else self.last_frame

    def shutdown(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        if self.enable_preview and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print("[CAM] Shutdown.")
