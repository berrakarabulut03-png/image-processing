# mapping/costmap.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional
import math
import time

import numpy as np
import cv2

# --- Parkur/arac sabitleri (sartname referansli) ---
ROAD_WIDTH_M = 3.0                 # 6.1
BARRIER_HEIGHT_M = 0.80             # 6.1
CONE_BASE_M = 0.40                  # 6.7
CONE_HEIGHT_M = 0.75                # 6.7
SLIDING_OBSTACLE_WIDTH_M = 1.0      # 6.8
SLIDING_OBSTACLE_SPEED_MPS = 0.20   # 6.8
STEP_OBSTACLE_HEIGHT_M = 0.15       # 6.6 dik engel
BUMP_HEIGHT_M = 0.05                # 6.9 tumsek
VEHICLE_WIDTH_MAX_M = 1.2           # 7.1


class ObstacleClass(Enum):
    BARRIER = "barrier"      # 6.1 - statik, tam blok
    CONE = "cone"            # 6.7 - statik, tam blok
    SLIDING = "sliding"      # 6.8 - dinamik, tam blok + ekstra pay
    CLIMBABLE = "climbable"  # 6.6 dik engel / 6.9 tumsek - engel degil, uzerinden gecilir
    UNKNOWN = "unknown"      # siniflandirilmamis ham nokta - guvenli varsayim: statik blok


@dataclass
class CostmapConfig:
    # planning/local_planner.py'deki VFH_LOOKAHEAD_M=10.0 ile eslesecek sekilde:
    # yaricap ~12 m (%20 emniyet payi) -> cap 24 m. res_m=0.10, 40 cm'lik koninin
    # (6.7) en az 4 hucreye dusmesini saglar (slalom hassasiyeti icin 0.20'den
    # daha guvenli - bkz. daha once tartisilan cozunurluk karari).
    world_size_m: float = 24.0   # map width/height in meters (centered on vehicle)
    res_m: float = 0.10          # meters per cell

    occ_cost: int = 100          # occupied cell cost (kesin engel: koni/bariyer - temas yasak)
    free_cost: int = 0           # free cell cost

    # Siniflandirilamamis ham lidar (UNKNOWN) maliyeti. occ_cost'tan DUSUK:
    # cim/tas/toz gibi seyler gereksiz yere "kesin blok" sayilip araci
    # dondurmasin. VFH hard-block esigi (95) altinda tutulur -> kuvvetle
    # kacinilir ama tek yol oysa gecilir. Sahada gercek lidar ile ince ayar.
    unknown_cost: int = 85       # siniflanamamis lidar noktasi (UNKNOWN)

    # inflation: grows obstacles with safety radius.
    # Turetme: yol 3 m (6.1) - en genis arac 1.2 m (7.1) => her yanda 0.9 m bosluk.
    # 0.6 m secilerek aracin ortalanmadan da manevra edebilecegi pay birakilir.
    inflation_radius_m: float = 0.6
    inflation_cost: int = 65     # cost at obstacle boundary (falls with distance)

    # Kayar engel (6.8) temassiz gecisi 50 puana bagli oldugundan, standart
    # inflation'a eklenen ekstra guvenlik payi.
    dynamic_obstacle_margin_m: float = 0.35

    # Statik katman (bariyer/koni/UNKNOWN) yavas soner: bunlar hareket etmez,
    # kisa sureli algi kaybinda (oklüzyon) haritadan silinmemeli. Yine de sensor
    # gurultusunu sonsuza kadar tutmamak icin cok yavas bir decay uygulanir.
    static_decay_per_sec: float = 8.0

    # Dinamik katman (kayar engel) hizla soner: engel 20 cm/s ile surekli hareket
    # eder (res=0.20 m -> ~1 hucre/s), eski konumdaki "hayalet" hucrelerin gercek
    # engel oradan ayrildiktan kisa sure sonra temizlenmesi gerekir.
    dynamic_decay_per_sec: float = 50.0


class Costmap2D:
    """
    Obstacle-class-aware local costmap (numpy + cv2 backend).

    Grid values: 0..100 (higher = more dangerous). Center of map = vehicle
    position (0,0). +X right, +Y forward.

    Iki ayri katman tutulur (farkli decay hizlari icin gerekli) ve okuma icin
    kullanilan `self.grid`, her ikisinin eleman-bazli maksimumu olarak guncel
    tutulur:
      - statik katman : BARRIER / CONE / UNKNOWN
      - dinamik katman: SLIDING
    CLIMBABLE siniflandirmasi hicbir katmana yazilmaz (asilmasi beklenen, yol
    olarak kabul edilen engel: dik engel 6.6, tumsek 6.9).
    """

    def __init__(self, cfg: Optional[CostmapConfig] = None):
        self.cfg = cfg or CostmapConfig()
        self.size = int(round(self.cfg.world_size_m / self.cfg.res_m))
        # ensure odd so center is exact cell
        if self.size % 2 == 0:
            self.size += 1
        self.c = self.size // 2

        self._static_grid = self._new_grid()
        self._dynamic_grid = self._new_grid()
        self.grid = self._new_grid()  # combined view consumed by VFH/streamer/renderer

        self._last_update_ts = time.time()

        # Kesirli sonme biriktiricileri: yuksek tik frekansinda (ör. 20+ Hz)
        # int(rate*dt) tek basina hep 0 dondugunden (ve _last_update_ts her
        # cagrida ilerledigi icin kesir bir daha geri gelmez), biriktirici
        # olmadan katman HIC sonmez. Bkz. decay().
        self._static_decay_accum: float = 0.0
        self._dynamic_decay_accum: float = 0.0

    def _new_grid(self) -> np.ndarray:
        return np.zeros((self.size, self.size), dtype=np.int16)

    def _recombine(self):
        np.maximum(self._static_grid, self._dynamic_grid, out=self.grid)

    # ---------- basic geometry ----------
    def world_to_grid(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        gx = int(round(x_m / self.cfg.res_m)) + self.c
        gy = self.c - int(round(y_m / self.cfg.res_m))  # y forward => smaller row
        if 0 <= gx < self.size and 0 <= gy < self.size:
            return gx, gy
        return None

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = (gx - self.c) * self.cfg.res_m
        y = (self.c - gy) * self.cfg.res_m
        return x, y

    def clear(self):
        self._static_grid = self._new_grid()
        self._dynamic_grid = self._new_grid()
        self.grid = self._new_grid()
        self._static_decay_accum = 0.0
        self._dynamic_decay_accum = 0.0

    # ---------- ego-motion ----------
    def shift_ego(self, dx_m: float, dy_m: float, dtheta_rad: float = 0.0):
        """Aracin kendi hareketine gore (odometry/telemetry: son tikten bu
        yana kat edilen dx,dy - arac govde ekseninde +X sag +Y ileri - ve
        donus dtheta) grid icerigini kaydirir/dondurur.

        Bu cagrilmadan decay()/add_points_xy() kullanildiginda, arac
        ilerledikce onceki tik'te yazilmis (henuz sonmemis) engel hucreleri
        SABIT grid indekslerinde kalir - yani arac-goreli konumlari yanlis
        olur (hayalet engel). Her tik basinda, decay()'den once cagrilmalidir.

        Not: Bu, eski (dogrulanmis) skaler per-pixel formulun numpy ile
        vektorlestirilmis halidir - cv2 affine/rotasyon isaret kurallarina
        guvenilmedi, matematik birebir korunup sadece hizlandirildi.
        """
        if dx_m == 0.0 and dy_m == 0.0 and dtheta_rad == 0.0:
            return
        self._static_grid = self._shift_grid(self._static_grid, dx_m, dy_m, dtheta_rad)
        self._dynamic_grid = self._shift_grid(self._dynamic_grid, dx_m, dy_m, dtheta_rad)
        self._recombine()

    def _shift_grid(self, grid: np.ndarray, dx_m: float, dy_m: float, dtheta_rad: float) -> np.ndarray:
        # Bu bir GATHER (backward-mapping): her HEDEF (yeni kare) hucresi icin,
        # o hucrenin degerini eski karede NEREDEN okuyacagini buluyoruz.
        #
        # Ileri (eski->yeni) donusum: new = R(-dtheta) * (old - (dx,dy))
        # (bu, tek-nokta ego-motion telafisinde kullanilan ayni formuldur, bkz.
        # onceki dogrulanmis skaler versiyon / obstacle_classifier.py).
        # Gather icin bunun TERSI gerekir: old = R(dtheta) * new + (dx,dy)
        # (R ortogonal oldugu icin R(-dtheta)^-1 = R(dtheta)).
        n = self.size
        res = self.cfg.res_m
        c = self.c

        gy_dst, gx_dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

        # grid_to_world, vektorlestirilmis (hedef hucrenin dunya koordinati)
        x = (gx_dst - c) * res
        y = (c - gy_dst) * res

        cos_t, sin_t = math.cos(dtheta_rad), math.sin(dtheta_rad)
        rx = x * cos_t - y * sin_t + dx_m
        ry = x * sin_t + y * cos_t + dy_m

        # world_to_grid, vektorlestirilmis (kaynak hucre indeksi)
        src_gx = np.round(rx / res).astype(np.int64) + c
        src_gy = c - np.round(ry / res).astype(np.int64)

        valid = (src_gx >= 0) & (src_gx < n) & (src_gy >= 0) & (src_gy < n)

        out = np.zeros_like(grid)
        out[gy_dst[valid], gx_dst[valid]] = grid[src_gy[valid], src_gx[valid]]
        return out

    # ---------- decay ----------
    def decay(self, now_ts: Optional[float] = None):
        now = now_ts or time.time()
        dt = max(0.0, now - self._last_update_ts)
        self._last_update_ts = now
        if dt <= 0:
            return

        # Kesirli miktarlari biriktir: esik 1'i gecene kadar kesir bir sonraki
        # tik'e tasinir, boylece yuksek frekansta bile katman zamanla dogru
        # sekilde soner (int(rate*dt) tek basina hep 0 donebilir).
        self._static_decay_accum += self.cfg.static_decay_per_sec * dt
        self._dynamic_decay_accum += self.cfg.dynamic_decay_per_sec * dt
        static_dec = int(self._static_decay_accum)
        dyn_dec = int(self._dynamic_decay_accum)
        if static_dec <= 0 and dyn_dec <= 0:
            return
        self._static_decay_accum -= static_dec
        self._dynamic_decay_accum -= dyn_dec

        if static_dec > 0:
            np.clip(self._static_grid - static_dec, 0, None, out=self._static_grid)
        if dyn_dec > 0:
            np.clip(self._dynamic_grid - dyn_dec, 0, None, out=self._dynamic_grid)
        self._recombine()

    # ---------- updates: classified perception input ----------
    def add_detection(self, det: Dict[str, Any]):
        """Feed one classified perception detection.

        Expected keys:
          'class': one of ObstacleClass values ('barrier'/'cone'/'sliding'/'climbable'),
                   missing/unrecognized -> ObstacleClass.UNKNOWN
          'x', 'y': vehicle-frame meters (same convention as world_to_grid:
                    +X right, +Y forward)
        """
        try:
            cls = ObstacleClass(str(det.get("class", "unknown")).lower())
        except ValueError:
            cls = ObstacleClass.UNKNOWN
        x = float(det.get("x", 0.0))
        y = float(det.get("y", 0.0))
        self.add_points_xy([(x, y)], obstacle_class=cls)

    def add_points_xy(
        self,
        points_xy: List[Tuple[float, float]],
        obstacle_class: ObstacleClass = ObstacleClass.UNKNOWN,
        base_cost: Optional[int] = None,
        extra_margin_m: Optional[float] = None,
        force_dynamic: bool = False,
    ):
        """Mark a batch of same-class points as occupied (+ inflation).

        BARRIER / CONE / UNKNOWN -> static layer (yavas soner), standard inflation
                                     (6.1 / 6.7: temas yok).
        SLIDING                  -> dynamic layer (hizli soner), standard inflation +
                                     dynamic_obstacle_margin_m (6.8: temassiz
                                     gecis 50 puan).
        CLIMBABLE                -> ignored entirely; vehicle is expected to
                                     drive over these (6.6 dik engel, 6.9
                                     tumsek), not steer around them.

        force_dynamic: sinifa bakmaksizin dinamik (hizli sonen) katmana yazar.
                        Kullanim alani: siniflandirilamayan (UNKNOWN) ama
                        OLCULEN hizi belirgin olan kumeler (beklenmedik hareketli
                        cisim). Boylece "hareket ettigini gordum ama ne oldugunu
                        bilmiyorum" bilgisi, statik katmanin yavas sonmesiyle
                        gereksiz uzun sureli "hayalet engel" birakmaz. CONE/BARRIER
                        icin KULLANILMAZ - bunlar bilinen sabit parkur elemanlari,
                        sensor gurultusuyle "hareket ediyor" sanilip erken
                        silinmemeli.
        """
        if obstacle_class == ObstacleClass.CLIMBABLE or not points_xy:
            return

        gx, gy = self._points_to_indices(points_xy)
        if gx.size == 0:
            return

        margin = extra_margin_m
        if margin is None:
            margin = self.cfg.dynamic_obstacle_margin_m if obstacle_class == ObstacleClass.SLIDING else 0.0
        radius_m = self.cfg.inflation_radius_m + max(0.0, margin)
        if base_cost is not None:
            occ = int(base_cost)
        elif obstacle_class == ObstacleClass.UNKNOWN:
            occ = int(self.cfg.unknown_cost)   # UNKNOWN daha dusuk maliyet
        else:
            occ = int(self.cfg.occ_cost)

        seed = np.zeros((self.size, self.size), dtype=np.uint8)
        seed[gy, gx] = 1
        inflated = self._inflate(seed, radius_m, occ)

        if obstacle_class == ObstacleClass.SLIDING or force_dynamic:
            np.maximum(self._dynamic_grid, inflated, out=self._dynamic_grid)
        else:
            np.maximum(self._static_grid, inflated, out=self._static_grid)
        self._recombine()

    def _points_to_indices(self, points_xy: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        gx = np.round(arr[:, 0] / self.cfg.res_m).astype(np.int64) + self.c
        gy = self.c - np.round(arr[:, 1] / self.cfg.res_m).astype(np.int64)
        valid = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)
        return gx[valid], gy[valid]

    def _inflate(self, seed_mask_u8: np.ndarray, radius_m: float, occ_cost: int) -> np.ndarray:
        """cv2.distanceTransform ile engel noktalarindan uzakliga gore azalan
        maliyet alani uretir. distanceTransform izotropik (yonsuz) oldugu icin
        donus/oteleme isaret belirsizligi tasimaz - eski per-nokta kernel
        yontemiyle matematiksel olarak esdegerdir, sadece vektorlestirilmistir.
        """
        # distanceTransform: 0 = "ozellik" (buradan mesafe olculur), >0 = arka plan
        inv = np.where(seed_mask_u8 > 0, 0, 255).astype(np.uint8)
        dist_cells = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        dist_m = dist_cells * self.cfg.res_m

        out = np.zeros(seed_mask_u8.shape, dtype=np.int16)
        if radius_m <= 1e-6:
            out[seed_mask_u8 > 0] = occ_cost
            return out

        within = dist_m <= radius_m
        t = 1.0 - (dist_m / radius_m)  # 1 at center, 0 at edge
        vals = self.cfg.inflation_cost + t * (occ_cost - self.cfg.inflation_cost)
        vals = np.clip(vals, 0, occ_cost)
        out[within] = vals[within].astype(np.int16)
        out[seed_mask_u8 > 0] = occ_cost
        return out

    # ---- convenience wrappers over add_points_xy ----
    def add_occupied_xy(
        self,
        points_xy: List[Tuple[float, float]],
        base_cost: Optional[int] = None,
        extra_margin_m: float = 0.0,
    ):
        """Raw/unclassified sensor points (e.g. plain lidar hits) -> ObstacleClass.UNKNOWN."""
        self.add_points_xy(
            points_xy,
            obstacle_class=ObstacleClass.UNKNOWN,
            base_cost=base_cost,
            extra_margin_m=(extra_margin_m or None),
        )

    def add_sliding_obstacle_xy(self, points_xy: List[Tuple[float, float]], base_cost: Optional[int] = None):
        """Convenience for the kayar engel (sartname 6.8)."""
        self.add_points_xy(points_xy, obstacle_class=ObstacleClass.SLIDING, base_cost=base_cost)

    def add_cone_xy(self, points_xy: List[Tuple[float, float]], base_cost: Optional[int] = None):
        """Convenience for trafik konileri (sartname 6.7)."""
        self.add_points_xy(points_xy, obstacle_class=ObstacleClass.CONE, base_cost=base_cost)

    def add_barrier_xy(self, points_xy: List[Tuple[float, float]], base_cost: Optional[int] = None):
        """Convenience for yol bariyerleri (sartname 6.1)."""
        self.add_points_xy(points_xy, obstacle_class=ObstacleClass.BARRIER, base_cost=base_cost)

    # ---------- output ----------
    def to_uint8(self):
        """Return grid as uint8 image (0..255), where higher = more occupied."""
        g = np.clip(self.grid.astype(np.float32), 0.0, 100.0)
        return (g * 2.55).astype(np.uint8)

    def serialize_packet(self) -> bytes:
        """Serialize costmap for UDP streaming to the ground station.
        Packet: magic(7)=BHCOST1, uint16 w,h, float32 res, float32 world, zlib(data)
        data is row-major uint8 (h*w)
        """
        img = self.to_uint8()
        if img is None:
            return b""
        h, w = img.shape
        payload = img.tobytes(order='C')
        comp = __import__('zlib').compress(payload, level=3)
        header = __import__('struct').pack('<7sHHff', b'BHCOST1', int(w), int(h), float(self.cfg.res_m), float(self.cfg.world_size_m))
        return header + comp
