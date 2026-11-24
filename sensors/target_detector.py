"""
Atis hedefi tespiti (sartname 6.10 / Sekil-5) - YOLO'suz.

Hedef: cerceveye oturtulmus A3 kagit uzerinde KONSANTRIK siyah halkalar
(6 cm ic daire, 12 cm ve 18 cm halkalar). Geometri sabit ve yuksek kontrastli
oldugu icin klasik goruntu isleme YOLO'dan hem hassas hem egitimsizdir:
nisan icin kaba kutu degil, ALT-PIKSEL MERKEZ gerekir.

Yontem (kontur tabanli - HoughCircles DEGIL):
  1) Griye cevir + blur + Otsu esikleme (siyah halkalar ikili maskeye)
  2) Konturlari bul; yeterince DAIRESEL olanlari sec
     (dairesellik = 4*pi*alan / cevre^2, 1'e yakin = daire)
  3) Ayni merkezi paylasan, FARKLI yaricapli >=2 dairesel kontur grubu ara
     - tek yuvarlak nesneler (koni tabani, teker) boylece elenir
  4) Grubun ortak merkezi = hedef merkezi
  Not: HoughCircles kullanilmadi cunku minDist parametresi ayni merkezli
  daireleri bilerek bastirir - konsantrik hedefle yapisal olarak celisir.

Cikti (detect):
  {"found": True, "cx": px, "cy": px,
   "err_x": goruntu merkezinden yatay sapma (px, + sag),
   "err_y": dikey sapma (px, + asagi),
   "radius_px": en buyuk halka yaricapi}
  ya da {"found": False}

Nisan dongusu (ileride, servo/pan-tilt donanimina gore):
  err_x/err_y -> servo komutu, |err| < esik olunca lazer ac >= 1 sn (6.10).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None


class TargetDetector:
    MIN_RADIUS_PX = 6          # bundan kucuk konturlar gurultu
    MIN_CIRCULARITY = 0.72     # 4*pi*A/P^2 (1 = mukemmel daire)
    CONCENTRIC_TOL_PX = 12     # merkezler bu kadar yakinsa "es merkezli" say
    MIN_RADIUS_RATIO = 1.35    # grup icindeki yaricaplar en az bu oranda farkli
                               # olmali (ayni cemberin ic/dis kenari sayilmasin)

    def detect(self, frame_bgr) -> Dict[str, Any]:
        """BGR karede hedefi ara; merkez + nisan hatasini dondur."""
        if not _HAS_CV2 or frame_bgr is None:
            return {"found": False}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # koyu halkalar on planda: Otsu + ters (siyah -> beyaz maske)
        _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cands: List[Tuple[float, float, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            per = cv2.arcLength(c, True)
            if per <= 0 or area < 30:
                continue
            circularity = 4.0 * np.pi * area / (per * per)
            if circularity < self.MIN_CIRCULARITY:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < self.MIN_RADIUS_PX:
                continue
            cands.append((float(x), float(y), float(r)))

        if len(cands) < 2:
            return {"found": False}

        # Konsantrik grup: es merkezli + BELIRGIN farkli yaricapli >= 2 daire
        best_group: List[Tuple[float, float, float]] = []
        for (x0, y0, _) in cands:
            group = [c for c in cands
                     if abs(c[0] - x0) <= self.CONCENTRIC_TOL_PX
                     and abs(c[1] - y0) <= self.CONCENTRIC_TOL_PX]
            radii = sorted(c[2] for c in group)
            # ayni kenarin ic/disi degil, gercek ayri halkalar mi?
            distinct = 1
            for i in range(1, len(radii)):
                if radii[i] / max(1e-6, radii[i - 1]) >= self.MIN_RADIUS_RATIO:
                    distinct += 1
            if distinct >= 2 and len(group) > len(best_group):
                best_group = group

        if len(best_group) < 2:
            return {"found": False}   # tek daire = hedef degil (koni/teker olabilir)

        cx = sum(c[0] for c in best_group) / len(best_group)
        cy = sum(c[1] for c in best_group) / len(best_group)
        r_max = max(c[2] for c in best_group)

        h, w = gray.shape[:2]
        return {
            "found": True,
            "cx": cx, "cy": cy,
            "err_x": cx - w / 2.0,   # + = hedef goruntunun saginda
            "err_y": cy - h / 2.0,   # + = hedef asagida
            "radius_px": r_max,
            "n_rings": len(best_group),
        }

    def draw(self, frame_bgr, det: Dict[str, Any]):
        """Tespiti kare uzerine cizer (debug/YKI onizleme)."""
        if not _HAS_CV2 or not det.get("found"):
            return frame_bgr
        out = frame_bgr.copy()
        cx, cy, r = int(det["cx"]), int(det["cy"]), int(det["radius_px"])
        cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(out, f"err=({det['err_x']:+.0f},{det['err_y']:+.0f})px",
                    (cx - 60, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        return out
