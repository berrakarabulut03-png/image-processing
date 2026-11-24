"""
Tabela tespit node'u - kamera + YOLO -> aktif parkur asamasi.

cone_detector.py'nin kardesi. FARK: tabela costmap'i beslemez (engel degil);
mission_manager'i besler - "su an hangi asama tabelasindayim?".

Cikti:
  process_detections() -> [{"sign", "conf", "box_frac", "bbox"}, ...]
  get_current_sign()   -> en guvenilir tek tabela (sign, conf) ya da None

Boyut filtresi (TABELA_MIN_BOX_FRAC): kucuk/uzak tabelalar yanlis okunabildigi
icin ( or. kucuk '11' -> '4') sadece yeterince buyuk (yani yeterince yakin)
tabelalar kabul edilir. Arac tabelaya yaklastikca zaten buyur.

Model yoksa/acilamazsa DUMMY moda duser (None/bos doner).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
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
        CAMERA_TABELA_MODEL_PATH, CAMERA_TABELA_CONF, CAMERA_TABELA_INDEX,
        TABELA_MIN_BOX_FRAC,
    )
except Exception:
    CAMERA_TABELA_MODEL_PATH = "models/tabela_best.pt"
    CAMERA_TABELA_CONF = 0.6
    CAMERA_TABELA_INDEX = 0
    TABELA_MIN_BOX_FRAC = 0.05

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Gecerli tabela sinif adlari (model 13 sinif uretir). Beklenmeyen ad atlanir.
VALID_SIGNS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "stop", "11_bitis"}


class TabelaDetector:
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
        self.camera_index = CAMERA_TABELA_INDEX if camera_index is None else int(camera_index)
        self.conf = CAMERA_TABELA_CONF if conf is None else float(conf)
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
            print("[TABELA] 'opencv' yok -> DUMMY.")
            return
        if not _HAS_YOLO:
            print("[TABELA] 'ultralytics' yok -> DUMMY.")
            return

        mp = Path(model_path or CAMERA_TABELA_MODEL_PATH)
        if not mp.is_absolute():
            mp = _PROJECT_ROOT / mp
        if not mp.exists():
            print(f"[TABELA] Model bulunamadi ({mp}) -> DUMMY.")
            return
        try:
            self.model = YOLO(str(mp))
            print(f"[TABELA] Model yuklendi: {mp} siniflar={self.model.names}")
        except Exception as e:
            print(f"[TABELA] Model yuklenemedi ({mp}): {e} -> DUMMY.")
            self.model = None
            return

        if not self._start_camera():
            print("[TABELA] Kamera acilamadi -> DUMMY.")
            self.model = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print("[TABELA] Hazir, reader thread basladi.")

    def _start_camera(self) -> bool:
        try:
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
                print(f"[TABELA] Reader hata: {e}")
                time.sleep(0.25)

    # ------------------------------------------------------------------ #
    def process_detections(self) -> List[Dict[str, Any]]:
        """Son karedeki tabela tespitleri (boyut filtresi uygulanmis)."""
        if self.model is None:
            return []
        with self._lock:
            frame = self._latest_frame
            ts = self._latest_ts
        if frame is None or (time.time() - ts) > self.timeout_s:
            return []
        return self.detect_in_image(frame)

    def detect_in_image(self, frame) -> List[Dict[str, Any]]:
        """Verilen BGR karede tabela tespiti (test icin de kullanilir)."""
        if self.model is None:
            return []
        try:
            r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        except Exception as e:
            print(f"[TABELA] Inference hata: {e}")
            return []

        img_h = frame.shape[0]
        dets: List[Dict[str, Any]] = []
        for box in r.boxes:
            sign = self.model.names[int(box.cls[0])]
            if sign not in VALID_SIGNS:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            box_frac = (y2 - y1) / img_h
            if box_frac < TABELA_MIN_BOX_FRAC:
                continue  # cok kucuk/uzak -> yanlis okuma riski, yok say
            dets.append({
                "sign": sign,
                "conf": float(box.conf[0]),
                "box_frac": float(box_frac),
                "bbox": [x1, y1, x2, y2],
            })

        try:
            overlay = r.plot()
            with self._lock:
                self._overlay_frame = overlay
            if self.enable_preview:
                cv2.imshow("Tabela", overlay)
                cv2.waitKey(1)
        except Exception:
            pass
        return dets

    def get_current_sign(self) -> Optional[Tuple[str, float]]:
        """En guvenilir TEK tabela: en buyuk (en yakin) olani sec.

        Donus: (sign, conf) ya da None. mission_manager bunu kullanir.
        """
        dets = self.process_detections()
        if not dets:
            return None
        best = max(dets, key=lambda d: d["box_frac"])  # en yakin = en buyuk kutu
        return (best["sign"], best["conf"])

    def is_healthy(self) -> bool:
        if self.model is None or not self._running:
            return False
        with self._lock:
            ts = self._latest_ts
        return ts > 0.0 and (time.time() - ts) <= max(2.0, 3.0 * self.timeout_s)

    def get_last_overlay_frame(self):
        with self._lock:
            return self._overlay_frame

    def get_last_frame(self):
        """Son HAM kare (kutu cizilmemis). Atis hedef tespiti bunu kullanir:
        overlay'deki YOLO kutulari halka tespitini bozabilir."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

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
        print("[TABELA] Shutdown.")
