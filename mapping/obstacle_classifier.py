# mapping/obstacle_classifier.py
"""
Lidar-only obstacle pre-classifier for the TEKNOFEST 2026 parkur sartnamesi.

Bu proje icinde koni/kayar-engel/bariyer ayrimi yapan bir kamera tabanli
nesne tanima (ornegin YOLO) hattı henuz yok - tek gercek zamanli geometri
kaynagi 2D lidar (sensors/lidar_node.py: sadece aci+mesafe, yukseklik yok).
Bu modul, costmap.py'nin ObstacleClass tabanli API'sini (add_points_xy) hicbir
kamera girdisine ihtiyac duymadan besleyebilmek icin SAF GEOMETRI + HAREKET
tabanli bir on-siniflandirma yapar:

  1) Ham tarama, komsu aci/mesafe farkina gore kumelere (cluster) ayrilir.
  2) Her kumenin genisligi (ilk-son nokta araligi), sartnamedeki bilinen
     engel olculeriyle karsilastirilir:
       - ~0.25-0.60 m  -> koni adayi                (6.7: 40 cm +-10 cm)
       - ~0.70-1.35 m  -> kayar engel adayi          (6.8: 1 m)
       - >= 1.4 m      -> bariyer/duvar              (6.1: surekli bariyer)
  3) Kume merkezleri ardisik tarama arasinda eslestirilip yanal hiz tahmin
     edilir. Costmap ARAC-MERKEZLI oldugundan, arac kendisi hareket ettiginde
     DURAN nesneler bile kareler arasi yer degistirmis gibi gorunur; bu yuzden
     onceki kume merkezleri, telemetriden gelen heading/groundspeed ile
     ego-hareket telafisi yapilarak simdiki kareye tasinir - ancak BUNDAN
     SONRA hiz hesaplanir.
  4) "Kayar engel adayi" genislikteki bir kume, ego-hareket telafisi
     SONRASINDA hala gercekci bir hizla (6.8: 0.20 m/s) hareket ediyor
     gorunuyorsa ObstacleClass.SLIDING olarak onaylanir (dinamik katman +
     ekstra guvenlik payi). Hareketi henuz dogrulanamayan (ilk kare, ego-motion
     bilgisi yok, veya duruyor gorunen) kayar-engel-genisligindeki kumeler
     guvenli tarafta kalinarak ObstacleClass.UNKNOWN'a duser: yine de
     statik+tam-blok muamele gorur, sadece ekstra payi almaz.

Bu bir ML/gorsel siniflandirici degildir; ileride kamera tabanli gercek bir
tanima eklenirse bu modulun yerini alabilir ya da onunla birlikte calisabilir
- ikisi de costmap.Costmap2D.add_points_xy(points, obstacle_class=...) ile
ayni sozlesmeyi kullanir.

Not: Bu dosya standalone'dir - hicbir yerden cagrilmiyor. planning/local_planner.py
(veya baska hicbir dosya) bilerek degistirilmedi; entegrasyon ayri bir istektir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import time

from mapping.costmap import ObstacleClass


@dataclass
class ClassifierConfig:
    max_range_m: float = 25.0           # bu menzilin disindaki lidar donusleri yok sayilir
    cluster_break_m: float = 0.30       # komsu nokta araligi bu esigi asarsa yeni kume baslar
    # RPLidar S3 ~0.11 deg/nokta yogunlugunda 5m'deki 40cm koni bile ~40 nokta
    # verir; 5, gurultu kumelerini elerken gercek engelleri kacirmaz.
    min_cluster_points: int = 5
    min_cluster_width_m: float = 0.08
    # S3 yogunlugunda tek kume yuzlerce nokta olabilir; costmap'e yazilan
    # Cluster.points bu sayiya uniform seyreltilir (genislik/siniflandirma
    # HER ZAMAN tam kumeden olculur, seyreltme yalnizca cikis noktalarini
    # etkiler). 0/None = seyreltme yok.
    max_points_per_cluster: int = 60

    # Lidar montaj yonu: lidar'in 0 derecesi arac ilerisi olmayabilir.
    # efektif aci = (+-ham aci + angle_offset_deg) % 360; varsayilanlar
    # (True, 0.0) mevcut davranisi birebir korur.
    angle_offset_deg: float = 0.0
    angle_clockwise: bool = True

    # sartname 6.7: koni 40 cm +-10 cm taban -> toleransli bant
    cone_width_lo_m: float = 0.25
    cone_width_hi_m: float = 0.60

    # sartname 6.8: kayar engel 1 m genislik -> toleransli bant
    sliding_width_lo_m: float = 0.70
    sliding_width_hi_m: float = 1.35

    # sartname 6.1: bundan genis/uzun kumeler surekli bariyer/duvar sayilir
    barrier_width_min_m: float = 1.40

    # ardisik tarama kume eslestirme ve kayar engel hiz onayi
    max_track_match_m: float = 0.6
    sliding_speed_lo_mps: float = 0.08  # 6.8: 0.20 m/s +- olcum/kare-hizi toleransi
    sliding_speed_hi_mps: float = 0.45


@dataclass
class Cluster:
    points: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    width_m: float
    obstacle_class: ObstacleClass
    est_speed_mps: Optional[float] = None
    est_velocity_xy: Optional[Tuple[float, float]] = None  # (vx, vy) m/s - kayar engel tahmini icin


class LidarObstacleClassifier:
    """Stateful: ardisik classify() cagrilari arasinda onceki kume
    merkezlerini ve zaman/heading bilgisini tutar (hiz tahmini icin).

    Kullanim (entegre edilmek istenirse, cagiran taraf sorumlulugunda):
        clf = LidarObstacleClassifier()
        clusters = clf.classify(lidar_data, forward_speed_mps=groundspeed, heading_deg=heading)
        for c in clusters:
            costmap.add_points_xy(c.points, obstacle_class=c.obstacle_class)
    """

    def __init__(self, cfg: Optional[ClassifierConfig] = None):
        self.cfg = cfg or ClassifierConfig()
        self._prev_centroids: List[Tuple[float, float]] = []
        self._prev_ts: Optional[float] = None
        self._prev_heading_deg: Optional[float] = None
        # Tekrarlanan tarama korumasi: lidar_node buffer'i ~5 Hz guncellenir
        # ama ana dongu 10 Hz okur -> ayni tarama iki kez gelebilir. Ayni
        # tarama ikinci kez islenirse yer degistirme 0 cikar, hiz tahmini
        # coker ve SLIDING onayi titrer. Parmak izi eslesirse onceki sonuc
        # dondurulur ve zaman/merkez state'i HIC guncellenmez.
        self._last_fingerprint: Optional[tuple] = None
        self._last_clusters: List[Cluster] = []

    def reset(self):
        self._prev_centroids = []
        self._prev_ts = None
        self._prev_heading_deg = None
        self._last_fingerprint = None
        self._last_clusters = []

    def classify(
        self,
        lidar_data: List[Tuple[float, float]],
        forward_speed_mps: Optional[float] = None,
        heading_deg: Optional[float] = None,
        now_ts: Optional[float] = None,
    ) -> List[Cluster]:
        """
        forward_speed_mps / heading_deg: aracin KENDI hareketi (telemetry'den,
        ornegin groundspeed + heading). Verilirse onceki kare kume merkezleri
        bu ego-hareketle telafi edilip SONRA hiz karsilastirmasi yapilir.
        Verilmezse hareket onayi devre disi kalir (guvenli varsayim uygulanir).
        """
        # Ayni buffer icerigi ikinci kez mi geldi? (bkz. __init__ notu)
        fp = self._scan_fingerprint(lidar_data)
        if fp is not None and fp == self._last_fingerprint:
            return list(self._last_clusters)
        self._last_fingerprint = fp

        now = now_ts if now_ts is not None else time.time()
        dt = None
        if self._prev_ts is not None:
            dt = max(1e-3, now - self._prev_ts)

        yaw_rate_radps = None
        if dt is not None and heading_deg is not None and self._prev_heading_deg is not None:
            dhead = self._wrap_deg(heading_deg - self._prev_heading_deg)
            yaw_rate_radps = math.radians(dhead) / dt

        ego_motion = None
        if dt is not None and forward_speed_mps is not None and yaw_rate_radps is not None:
            ego_motion = (0.0, float(forward_speed_mps), float(yaw_rate_radps))

        compensated_prev = self._prev_centroids
        if ego_motion is not None and self._prev_centroids:
            compensated_prev = self._compensate_ego_motion(self._prev_centroids, ego_motion, dt)

        pts = self._to_xy_sorted_by_angle(lidar_data)
        raw_clusters = self._cluster(pts)

        clusters: List[Cluster] = []
        new_centroids: List[Tuple[float, float]] = []

        for cpts in raw_clusters:
            if len(cpts) < self.cfg.min_cluster_points:
                continue
            width = self._cluster_width(cpts)
            if width < self.cfg.min_cluster_width_m:
                continue

            centroid = (
                sum(p[0] for p in cpts) / len(cpts),
                sum(p[1] for p in cpts) / len(cpts),
            )

            speed = None
            velocity = None
            if dt is not None:
                velocity = self._nearest_velocity(centroid, compensated_prev, dt)
                if velocity is not None:
                    speed = math.hypot(velocity[0], velocity[1])

            obstacle_class = self._classify(width, speed)

            clusters.append(Cluster(
                points=self._downsample(cpts),
                centroid=centroid,
                width_m=width,
                obstacle_class=obstacle_class,
                est_speed_mps=speed,
                est_velocity_xy=velocity,
            ))
            new_centroids.append(centroid)

        self._prev_centroids = new_centroids
        self._prev_ts = now
        if heading_deg is not None:
            self._prev_heading_deg = heading_deg
        self._last_clusters = clusters
        return clusters

    def _downsample(self, cpts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """S3 yogunlugundaki kumeleri costmap yuku icin seyreltir; ilk/son
        nokta (kumenin uclari) her zaman korunur. Genislik olcumu ve
        siniflandirma bu fonksiyondan ONCE, tam kume uzerinden yapilir."""
        n_max = int(self.cfg.max_points_per_cluster or 0)
        n = len(cpts)
        if n_max <= 0 or n <= n_max:
            return cpts
        step = (n - 1) / (n_max - 1)
        return [cpts[round(i * step)] for i in range(n_max)]

    @staticmethod
    def _scan_fingerprint(lidar_data: List[Tuple[float, float]]) -> Optional[tuple]:
        try:
            n = len(lidar_data)
            if n == 0:
                return None
            head = tuple(tuple(p) for p in lidar_data[:3])
            tail = tuple(tuple(p) for p in lidar_data[-3:])
            return (n, head, tail)
        except Exception:
            return None

    # ---------- classification ----------
    def _classify(self, width_m: float, speed_mps: Optional[float]) -> ObstacleClass:
        cfg = self.cfg
        if width_m >= cfg.barrier_width_min_m:
            return ObstacleClass.BARRIER
        if cfg.cone_width_lo_m <= width_m <= cfg.cone_width_hi_m:
            return ObstacleClass.CONE
        if cfg.sliding_width_lo_m <= width_m <= cfg.sliding_width_hi_m:
            if speed_mps is not None and cfg.sliding_speed_lo_mps <= speed_mps <= cfg.sliding_speed_hi_mps:
                return ObstacleClass.SLIDING
            # genislik uyuyor ama hareket henuz dogrulanamadi (ilk kare / durgun
            # gorunuyor) -> guvenli tarafta kal: statik+tam-blok, sadece ekstra
            # dinamik payi verme.
            return ObstacleClass.UNKNOWN
        return ObstacleClass.UNKNOWN

    # ---------- geometry ----------
    def _to_xy_sorted_by_angle(self, lidar_data) -> List[Tuple[float, float]]:
        offset = float(self.cfg.angle_offset_deg)
        cw = bool(self.cfg.angle_clockwise)
        tagged = []
        for ang_deg, d_m in lidar_data:
            try:
                d = float(d_m)
                if d <= 0.05 or d > self.cfg.max_range_m:
                    continue
                # montaj donusumu: efektif aci = (+-ham + ofset), arac govde
                # ekseninde 0 = ileri, saat yonu pozitif (x=sin, y=cos)
                raw = float(ang_deg)
                a = ((raw if cw else -raw) + offset) % 360.0
                rad = math.radians(a)
                x = d * math.sin(rad)
                y = d * math.cos(rad)
                tagged.append((a, x, y))
            except Exception:
                continue
        tagged.sort(key=lambda p: p[0])
        return [(x, y) for (_, x, y) in tagged]

    def _cluster(self, pts: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        clusters: List[List[Tuple[float, float]]] = []
        cur: List[Tuple[float, float]] = []
        prev = None
        for p in pts:
            if prev is not None:
                d = math.hypot(p[0] - prev[0], p[1] - prev[1])
                if d > self.cfg.cluster_break_m and cur:
                    clusters.append(cur)
                    cur = []
            cur.append(p)
            prev = p
        if cur:
            clusters.append(cur)

        # 360 derece sarma birlestirmesi: noktalar aci sirali oldugundan, tam
        # ondeki (aci ~358 -> ~2) bir nesne listenin basina ve sonuna bolunur;
        # 40cm'lik koni iki ~20cm kumeye ayrilip koni bandinin altinda kalirdi.
        # Ilk kumenin ilk noktasi ile son kumenin son noktasi fiziksel olarak
        # komsuysa iki kumeyi tek kume olarak birlestir (sira: son + ilk, ki
        # uclardan genislik olcumu dogru ciksin).
        if len(clusters) >= 2:
            first_pt = clusters[0][0]
            last_pt = clusters[-1][-1]
            d_wrap = math.hypot(first_pt[0] - last_pt[0], first_pt[1] - last_pt[1])
            if d_wrap <= self.cfg.cluster_break_m:
                merged = clusters[-1] + clusters[0]
                clusters = [merged] + clusters[1:-1]
        return clusters

    @staticmethod
    def _cluster_width(cpts: List[Tuple[float, float]]) -> float:
        # Noktalar aci sirali/komsu oldugundan ilk-son nokta araligi, nesnenin
        # sensore donuk yuzunun genisligini yeterli dogrulukta yaklastirir.
        if len(cpts) < 2:
            return 0.0
        x0, y0 = cpts[0]
        x1, y1 = cpts[-1]
        return math.hypot(x1 - x0, y1 - y0)

    def _nearest_velocity(
        self,
        centroid: Tuple[float, float],
        prev_centroids: List[Tuple[float, float]],
        dt: float,
    ) -> Optional[Tuple[float, float]]:
        """Eslesen (en yakin) onceki merkeze gore hiz VEKTORU (vx, vy) m/s.

        Skaler hiz buradan turetilir (buyukluk). Yon, kayar engelin (6.8)
        gelecekteki konumunu tahmin etmek icin gereklidir.
        """
        if not prev_centroids:
            return None
        best_d = None
        best_pc = None
        for pc in prev_centroids:
            d = math.hypot(centroid[0] - pc[0], centroid[1] - pc[1])
            if d <= self.cfg.max_track_match_m and (best_d is None or d < best_d):
                best_d = d
                best_pc = pc
        if best_pc is None:
            return None
        return ((centroid[0] - best_pc[0]) / dt, (centroid[1] - best_pc[1]) / dt)

    @staticmethod
    def _compensate_ego_motion(
        centroids: List[Tuple[float, float]],
        ego_motion: Tuple[float, float, float],
        dt: float,
    ) -> List[Tuple[float, float]]:
        """Onceki kare merkezlerini, aracin kendi hareketine (vx,vy,yaw_rate,
        arac govde ekseninde) gore simdiki arac-merkezli karaye tasir."""
        vx, vy, yaw_rate = ego_motion
        dx = vx * dt
        dy = vy * dt
        dtheta = yaw_rate * dt
        cos_t, sin_t = math.cos(-dtheta), math.sin(-dtheta)
        out = []
        for (x, y) in centroids:
            sx, sy = x - dx, y - dy
            rx = sx * cos_t - sy * sin_t
            ry = sx * sin_t + sy * cos_t
            out.append((rx, ry))
        return out

    @staticmethod
    def _wrap_deg(a: float) -> float:
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a
