# comm/goruntu_stream.py
"""
Kamera goruntusunu UDP ile yer istasyonuna aktarma katmani (sartname 6.12).

Neden: manuel kosuda pilot arac yaninda yuruyemez, KAMERADAN surer. Kameralar
aractaki Jetson'da; pilot ekrani YKI bilgisayarinda. Aradaki koprü LiteBeam
5AC (seffaf IP koprusu) - bu modul goruntuyu o kopruden tasiyan yazilimdir.

Tasarim:
  - Kare -> kucult (GORUNTU_GENISLIK) -> JPEG sikistir -> TEK UDP paketi.
    Tek paket = parcalama/birlestirme yok = basit ve saglam. JPEG 640px'te
    ~15-35 KB, UDP siniri ~64 KB'nin rahat altinda. Paket buyurse kalite
    otomatik dusurulur; yine sigmarsa kare atlanir (canli yayinda kayip kare
    sorun degil, GECIKME sorundur).
  - UDP secildi (TCP degil): kaybolan kare geri istenmez, en taze kare her
    seydir. Pilot 1 sn onceki goruntuyle suremez.
  - Paket basligi: magic BHGOR1 + kamera_id + isim(8) + gonderim zamani.
    Ayni porttan N kamera akar; alici id'ye gore ayirir.
  - Gecikme notu: JPEG yolu loopback'te ~10-40 ms, LiteBeam ustunde +5-20 ms
    beklenir. Jetson'a gecince istenirse GStreamer+NVENC (donanim sikistirma)
    hattina yukseltilebilir - protokol ayni kalir, sadece tasiyici degisir.

Kullanim: goruntu_verici.py (arac tarafi CLI) ve pilot_ekrani.py (YKI tarafi
CLI) bu siniflari sarar. Testler: scratchpad/test_goruntu_stream.py (loopback).
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2

try:
    from config import (GORUNTU_HEDEF_IP, GORUNTU_UDP_PORT,
                        GORUNTU_GENISLIK, GORUNTU_JPEG_KALITE)
except Exception:
    GORUNTU_HEDEF_IP = "127.0.0.1"
    GORUNTU_UDP_PORT = 14700
    GORUNTU_GENISLIK = 640
    GORUNTU_JPEG_KALITE = 60

_MAGIC = b"BHGOR1"
_BASLIK = "<6sB8sd"                     # magic, kamera_id, isim, gonderim_ts
_BASLIK_BOY = struct.calcsize(_BASLIK)  # 23 bayt
_MAX_PAKET = 60000                      # UDP guvenli tavan (bayt)


def paket_kur(kamera_id: int, isim: str, ts: float, jpeg: bytes) -> bytes:
    ad = isim.encode("ascii", errors="ignore")[:8].ljust(8, b"\x00")
    return struct.pack(_BASLIK, _MAGIC, int(kamera_id) & 0xFF, ad, float(ts)) + jpeg


def paket_coz(veri: bytes) -> Optional[Tuple[int, str, float, bytes]]:
    """(kamera_id, isim, gonderim_ts, jpeg) ya da None (yabanci/bozuk paket)."""
    if len(veri) <= _BASLIK_BOY:
        return None
    magic, kid, ad, ts = struct.unpack_from(_BASLIK, veri)
    if magic != _MAGIC:
        return None
    isim = ad.rstrip(b"\x00").decode("ascii", errors="ignore")
    return int(kid), isim, float(ts), veri[_BASLIK_BOY:]


class GoruntuVerici:
    """Arac (Jetson) tarafi: kareleri sikistirip UDP ile yollar."""

    def __init__(self, hedef_ip: Optional[str] = None, port: Optional[int] = None,
                 genislik: Optional[int] = None, kalite: Optional[int] = None):
        self.hedef = (hedef_ip or GORUNTU_HEDEF_IP, int(port or GORUNTU_UDP_PORT))
        self.genislik = int(genislik or GORUNTU_GENISLIK)
        self.kalite = int(kalite or GORUNTU_JPEG_KALITE)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1_000_000)
        self.gonderilen = 0
        self.atlanan = 0

    def gonder(self, kamera_id: int, isim: str, kare) -> bool:
        """Tek kareyi yolla. Buyukse kaliteyi dusur; yine sigmarsa atla."""
        if kare is None:
            return False
        h, w = kare.shape[:2]
        if w > self.genislik:
            olcek = self.genislik / float(w)
            kare = cv2.resize(kare, (self.genislik, int(h * olcek)))

        for kalite in (self.kalite, max(20, self.kalite - 25)):
            ok, jpeg = cv2.imencode(".jpg", kare,
                                    [cv2.IMWRITE_JPEG_QUALITY, int(kalite)])
            if not ok:
                self.atlanan += 1
                return False
            if len(jpeg) + _BASLIK_BOY <= _MAX_PAKET:
                try:
                    self._sock.sendto(paket_kur(kamera_id, isim, time.time(),
                                                jpeg.tobytes()), self.hedef)
                    self.gonderilen += 1
                    return True
                except Exception:
                    self.atlanan += 1
                    return False
        self.atlanan += 1   # iki kalitede de sigmadi (olagandisi buyuk kare)
        return False

    def kapat(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class GoruntuAlici:
    """YKI tarafi: UDP paketlerini arka planda toplar, kamera basina en taze
    kareyi tutar. pilot_ekrani.py bunu cizer."""

    def __init__(self, port: Optional[int] = None):
        self.port = int(port or GORUNTU_UDP_PORT)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4_000_000)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(0.2)
        self._kilit = threading.Lock()
        # kamera_id -> {"isim", "kare", "gonderim_ts", "alim_ts", "sayac"}
        self._kanallar: Dict[int, Dict[str, Any]] = {}
        self._calisiyor = True
        self._is = threading.Thread(target=self._dongu, daemon=True)
        self._is.start()

    def _dongu(self) -> None:
        while self._calisiyor:
            try:
                veri, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception:
                if self._calisiyor:
                    time.sleep(0.05)
                continue
            coz = paket_coz(veri)
            if coz is None:
                continue
            kid, isim, ts, jpeg = coz
            kare = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                                cv2.IMREAD_COLOR)
            if kare is None:
                continue
            with self._kilit:
                k = self._kanallar.setdefault(kid, {"sayac": 0})
                k["isim"] = isim or f"K{kid}"
                k["kare"] = kare
                k["gonderim_ts"] = ts
                k["alim_ts"] = time.time()
                k["sayac"] += 1

    def kanallar(self) -> Dict[int, Dict[str, Any]]:
        """Kopya dondurur: {id: {isim, kare, gonderim_ts, alim_ts, sayac}}."""
        with self._kilit:
            return {kid: dict(k) for kid, k in self._kanallar.items()}

    def kapat(self) -> None:
        self._calisiyor = False
        try:
            self._is.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
