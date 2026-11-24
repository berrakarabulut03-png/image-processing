# mapping/occupancy_map.py
from __future__ import annotations

import math
from typing import Iterable, Tuple, List

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class OccupancyMapView(QtWidgets.QGraphicsView):
    """
    Robot süpürge mantığı:
    - Arac merkezli local occupancy map
    - Engeller birikir, zamanla sönümlenir (decay)
    - Scan noktaları (x,y metre) ile beslenir
    """
    def __init__(
        self,
        parent=None,
        size_px: int = 420,
        res_m: float = 0.10,         # 10 cm / hücre
        world_size_m: float = 40.0,  # 40m x 40m
        decay_per_tick: int = 2,     # her tick'te yoğunluk azaltma
        tick_ms: int = 50
    ):
        super().__init__(parent)

        self.setRenderHints(
            QtGui.QPainter.Antialiasing |
            QtGui.QPainter.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)

        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(7, 10, 18)))
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)

        self.res_m = float(res_m)
        self.world_size_m = float(world_size_m)
        self.decay_per_tick = int(decay_per_tick)

        # grid dims
        self.n = int(self.world_size_m / self.res_m)
        self.n = max(64, min(self.n, 1200))
        if self.n % 2 != 0:
            self.n += 1

        self.cx = self.n // 2
        self.cy = self.n // 2

        # intensity grid 0..255
        self.grid = np.zeros((self.n, self.n), dtype=np.uint8)

        # display item
        self.pix_item = QtWidgets.QGraphicsPixmapItem()
        self.pix_item.setZValue(0)
        self.scene.addItem(self.pix_item)

        # overlay grid/rings
        self._overlay_item = QtWidgets.QGraphicsPathItem()
        self._overlay_item.setZValue(5)
        self.scene.addItem(self._overlay_item)

        # zoom bookkeeping
        self._user_zoom = 1.0
        self._auto_fit = True

        self._rebuild_scene_rect()
        self._build_overlay()

        # tick timer (decay + render)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(tick_ms))

        self._dirty = True

    def reset_view(self):
        self.resetTransform()
        self._user_zoom = 1.0
        self._auto_fit = True
        self._rebuild_scene_rect()
        self._dirty = True

    def clear_map(self):
        self.grid.fill(0)
        self._dirty = True

    def add_points_xy(self, points_m: Iterable[Tuple[float, float]]):
        """
        points_m: iterable of (x,y) in meters, where +y forward/up, +x right
        """
        pts = list(points_m)
        if not pts:
            return

        # Paint with small brush
        brush_r = 1  # cells
        inc = 35     # intensity increment

        for x_m, y_m in pts:
            gx = int(self.cx + (x_m / self.res_m))
            gy = int(self.cy - (y_m / self.res_m))
            if gx < 0 or gy < 0 or gx >= self.n or gy >= self.n:
                continue

            x0 = max(0, gx - brush_r)
            x1 = min(self.n, gx + brush_r + 1)
            y0 = max(0, gy - brush_r)
            y1 = min(self.n, gy + brush_r + 1)

            # saturating add
            patch = self.grid[y0:y1, x0:x1].astype(np.uint16) + inc
            self.grid[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

        self._dirty = True

    def _tick(self):
        # decay
        if self.decay_per_tick > 0:
            g = self.grid
            # subtract with saturation
            self.grid = np.where(g > self.decay_per_tick, g - self.decay_per_tick, 0).astype(np.uint8)

        # render if needed
        if self._dirty:
            self._render()
            self._dirty = False

    def _render(self):
        # Build RGBA image: blue-cyan intensity on dark background
        g = self.grid

        # normalize to 0..1
        a = (g.astype(np.float32) / 255.0)

        # color: cyan-ish
        r = (40 + 120 * a).astype(np.uint8)
        b = (80 + 170 * a).astype(np.uint8)
        gr = (60 + 170 * a).astype(np.uint8)

        alpha = (20 + 220 * (a ** 0.9)).astype(np.uint8)

        rgba = np.dstack([r, gr, b, alpha])  # H,W,4

        h, w, _ = rgba.shape
        qimg = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        pm = QtGui.QPixmap.fromImage(qimg.copy())  # copy to detach from numpy buffer

        self.pix_item.setPixmap(pm)
        self.pix_item.setOffset(-w/2, -h/2)

        # fit once
        if self._auto_fit:
            self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self._auto_fit = False

    def _rebuild_scene_rect(self):
        side = float(self.n)
        self.scene.setSceneRect(-side/2, -side/2, side, side)

    def _build_overlay(self):
        # draw rings + axes in pixel space (cells)
        side = float(self.n)
        radius = side * 0.46

        pen = QtGui.QPen(QtGui.QColor(45, 90, 150, 160))
        pen.setWidthF(1.1)

        p = QtGui.QPainterPath()

        # rings for 5m/10m/20m etc (in cells)
        for rm in [5, 10, 15, 20]:
            rr = (rm / self.res_m)
            if rr > radius:
                continue
            p.addEllipse(QtCore.QPointF(0, 0), rr, rr)

        # axes
        p.moveTo(-radius, 0); p.lineTo(radius, 0)
        p.moveTo(0, -radius); p.lineTo(0, radius)

        self._overlay_item.setPath(p)
        self._overlay_item.setPen(pen)
        self._overlay_item.setBrush(QtCore.Qt.NoBrush)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.angleDelta().y() == 0:
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        new_zoom = self._user_zoom * factor
        if new_zoom < 0.2 or new_zoom > 8.0:
            return
        self._user_zoom = new_zoom
        self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # do not kill user zoom
        if self._auto_fit:
            self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self._auto_fit = False
