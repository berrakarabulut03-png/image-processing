"""
Kamera testi - koni ve tabela tespitinin CANLI dogrulamasi.

Kullanim (arac kamerasi takiliyken ya da laptopta):
    python kamera_test.py                # once tabela 15sn, sonra koni 15sn
    python kamera_test.py --mode koni    # sadece koni
    python kamera_test.py --mode tabela  # sadece tabela
    python kamera_test.py --sure 30      # her modeli 30'ar sn test et

Ekranda canli onizleme penceresi acilir (kutular + sinif + guven), terminale
tespitler yazilir. Kapatmak icin: sure dolmasini bekle ya da Ctrl+C.

Not: Iki detector ayni kamerayi AYNI ANDA acamaz (isletim sistemi kisiti);
bu yuzden test SIRALI calisir. Aracta koni/tabela farkli kameralardaysa
config.py'de CAMERA_INDEX / CAMERA_TABELA_INDEX ayarla.
"""

from __future__ import annotations

import argparse
import time


def test_tabela(sure_s: float) -> None:
    from sensors.tabela_detector import TabelaDetector
    print(f"\n===== TABELA TESTI ({sure_s:.0f} sn) - pencereye tabela goster =====")
    det = TabelaDetector(enable_preview=True)
    t0 = time.time()
    try:
        while time.time() - t0 < sure_s:
            dets = det.process_detections()
            if dets:
                s = ", ".join(f"{d['sign']}(%{d['conf']*100:.0f})" for d in dets)
                print(f"  tabela: {s}")
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        det.shutdown()


def test_koni(sure_s: float) -> None:
    from sensors.cone_detector import ConeDetector
    print(f"\n===== KONI TESTI ({sure_s:.0f} sn) - pencereye koni goster =====")
    det = ConeDetector(enable_preview=True)
    t0 = time.time()
    try:
        while time.time() - t0 < sure_s:
            dets = det.process_detections()
            if dets:
                s = ", ".join(
                    f"koni(%{d['conf']*100:.0f}, {d['angle_deg']:+.0f}deg, "
                    f"{d['distance_m']:.1f}m)" for d in dets
                )
                print(f"  {s}")
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        det.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tabela", "koni", "hepsi"], default="hepsi")
    ap.add_argument("--sure", type=float, default=15.0, help="her model icin sn")
    a = ap.parse_args()

    if a.mode in ("tabela", "hepsi"):
        test_tabela(a.sure)
    if a.mode in ("koni", "hepsi"):
        test_koni(a.sure)
    print("\nTest bitti.")
