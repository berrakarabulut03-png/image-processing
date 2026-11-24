"""
Koni tespit node'u - kamera + YOLO -> costmap beslemesi.

sensors/camera_node.py kalibindan uyarlandi (reader thread, DUMMY
fallback, guvenilmez mesafe isaretleme). Cikti sozlesmesi, planner'daki
_fuse_camera_cones'un bekledigi formattir:

    camera_data = {"detections": cone_detector.process_detections()}
    dets = [
      {"class": "cone", "conf": 0.93,
       "angle_deg": +sag/-sol (derece),
       "distance_m": pinhole kestirimi (metre),
       "distance_src": "pinhole" | "" (guvenilmezse bos, distance_m=0),
       "bbox": [x1, y1, x2, y2]},
      ...
    ]

Mesafe su an PINHOLE kestirimi (sartname 6.7: koni tabani 0.40 m sabitinden).
Derinlik kamerasi (RealSense vb.) takilirsa camera_node'daki depth yolu
buraya da eklenebilir - planner tarafi distance_src alanini zaten taniyor.

Model yoksa/acilamazsa DUMMY moda duser (bos liste doner, sistem cokmez).
Yeni koni modeli hazir olunca models/cone_best.pt olarak koymak yeterli.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import sys
import threading
import time
from pathlib import Path

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False
    YOLO = None

try:
    from config import (
        CAMERA_CONE_MODEL_PATH, CAMERA_CONE_CONF, CAMERA_INDEX,
        CAMERA_HFOV_DEG, CONE_WIDTH_M,
    )
except Exception:
    CAMERA_CONE_MODEL_PATH = "models/cone_best.pt"
    CAMERA_CONE_CONF = 0.4
    CAMERA_INDEX = 0
    CAMERA_HFOV_DEG = 66.0
    CONE_WIDTH_M = 0.40

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model sinif adi -> sistem sinifi. Iki koni modelimiz de desteklenir:
# robotica ('cone') ve cone.v3 ('traffic_cone'). Taninmayan sinif atlanir.
CLASS_MAP = {"cone": "cone", "traffic_cone": "cone"}

# Pinhole sonucu bundan buyukse guvenilmez say (distance_src="", distance_m=0)
MAX_SANE_DISTANCE_M = 25.0


class ConeDetector:
    def __init__(
        self,
        enable_preview: bool = False,
        timeout_ms: int = 700,
        warmup_frames: int = 30,
        camera_index: Optional[int] = None,
        model_path: Optional[str] = None,
        conf: Optional[float] = None,
        imgsz: int = 640,
    ):
        self.enable_preview = bool(enable_preview)
        self.timeout_s = float(timeout_ms) / 1000.0
        self.warmup_frames = int(warmup_frames)
        self.camera_index = CAMERA_INDEX if camera_index is None else int(camera_index)
        self.conf = CAMERA_CONE_CONF if conf is None else float(conf)
        self.imgsz = int(imgsz)

        self.model = None
        self.cap = None

        self._latest_frame = None
        self._latest_ts = 0.0
        self._overlay_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if not _HAS_CV2:
            print("[CONE] 'opencv' yok -> DUMMY.")
            return
        if not _HAS_YOLO:
            print("[CONE] 'ultralytics' yok -> DUMMY.")
            return

        mp = Path(model_path or CAMERA_CONE_MODEL_PATH)
        if not mp.is_absolute():
            mp = _PROJECT_ROOT / mp
        if not mp.exists():
            print(f"[CONE] Model bulunamadi ({mp}) -> DUMMY. "
                  "Yeni model hazir olunca models/cone_best.pt olarak koy.")
            return
        try:
            self.model = YOLO(str(mp))
            print(f"[CONE] Model yuklendi: {mp} siniflar={self.model.names}")
            unknown = [n for n in self.model.names.values() if n not in CLASS_MAP]
            if unknown:
                print(f"[CONE] UYARI: CLASS_MAP disinda siniflar: {unknown} -> bu tespitler atlanir")
        except Exception as e:
            print(f"[CONE] Model yuklenemedi ({mp}): {e} -> DUMMY.")
            self.model = None
            return

        if not self._start_camera():
            print("[CONE] Kamera acilamadi -> DUMMY.")
            self.model = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print("[CONE] Hazir, reader thread basladi.")

    # ------------------------------------------------------------------ #
    def _start_camera(self) -> bool:
        try:
            # DSHOW Windows'a ozgu; Linux/Jetson'da backend'i OpenCV secsin
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            return bool(self.cap.isOpened())
        except Exception:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            return False

    def _reader_loop(self) -> None:
        frame_i = 0
        while self._running and self.cap is not None:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                frame_i += 1
                if frame_i <= self.warmup_frames:
                    continue
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.time()
                time.sleep(0.005)
            except Exception as e:
                print(f"[CONE] Reader hata: {e}")
                time.sleep(0.25)

    # ------------------------------------------------------------------ #
    # Ana API
    # ------------------------------------------------------------------ #
    def process_detections(self) -> List[Dict[str, Any]]:
        """Son kamera karesindeki koni tespitleri (planner sozlesmesi)."""
        if self.model is None:
            return []
        with self._lock:
            frame = self._latest_frame
            ts = self._latest_ts
        if frame is None or (time.time() - ts) > self.timeout_s:
            return []
        return self.detect_in_image(frame)

    def detect_in_image(self, frame) -> List[Dict[str, Any]]:
        """Verilen BGR karede koni tespiti yapar (test/kayit icin de kullanilir)."""
        if self.model is None:
            return []
        try:
            r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        except Exception as e:
            print(f"[CONE] Inference hata: {e}")
            return []

        img_w = frame.shape[1]
        focal = (img_w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)

        dets: List[Dict[str, Any]] = []
        for box in r.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            if cls_name not in CLASS_MAP:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            bw = max(1.0, x2 - x1)
            cx = (x1 + x2) / 2.0

            angle_deg = ((cx / img_w) - 0.5) * CAMERA_HFOV_DEG
            distance_m = (CONE_WIDTH_M * focal) / bw
            distance_src = "pinhole"
            # kutu kadraj kenarinda kirpiksa genislik kuculur -> mesafe siser
            if x1 < 2.0 or x2 > img_w - 2.0 or distance_m > MAX_SANE_DISTANCE_M:
                distance_src = ""
                distance_m = 0.0

            dets.append({
                "class": CLASS_MAP[cls_name],
                "conf": float(box.conf[0]),
                "angle_deg": float(angle_deg),
                "distance_m": float(distance_m),
                "distance_src": distance_src,
                "bbox": [x1, y1, x2, y2],
            })

        try:
            overlay = r.plot()
            with self._lock:
                self._overlay_frame = overlay
            if self.enable_preview:
                cv2.imshow("Cone", overlay)
                cv2.waitKey(1)
        except Exception:
            pass
        return dets

    def is_healthy(self) -> bool:
        """Kamera akisi canli mi? (DUMMY ya da akis kesikse False)."""
        if self.model is None or not self._running:
            return False
        with self._lock:
            ts = self._latest_ts
        return ts > 0.0 and (time.time() - ts) <= max(2.0, 3.0 * self.timeout_s)

    def get_last_overlay_frame(self):
        with self._lock:
            return self._overlay_frame

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if _HAS_CV2 and self.enable_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[CONE] Shutdown.")
