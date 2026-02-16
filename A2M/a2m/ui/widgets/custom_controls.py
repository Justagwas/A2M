from __future__ import annotations
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyleOptionButton, QStyleOptionSlider, QProxyStyle, QWidget

class RoundHandleSliderStyle(QProxyStyle):

    def __init__(self, *, handle_color: str, border_color: str, groove_color: str, fill_color: str, handle_size: int=18, groove_height: int=6, parent: QWidget | None=None) -> None:
        super().__init__(parent.style() if parent is not None else None)
        self._handle_color = QColor(handle_color)
        self._border_color = QColor(border_color)
        self._groove_color = QColor(groove_color)
        self._fill_color = QColor(fill_color)
        self._handle_size = max(12, int(handle_size))
        self._groove_height = max(4, int(groove_height))

    def set_colors(self, *, handle_color: str, border_color: str, groove_color: str, fill_color: str) -> None:
        self._handle_color = QColor(handle_color)
        self._border_color = QColor(border_color)
        self._groove_color = QColor(groove_color)
        self._fill_color = QColor(fill_color)

    def set_metrics(self, *, handle_size: int, groove_height: int) -> None:
        self._handle_size = max(12, int(handle_size))
        self._groove_height = max(4, int(groove_height))

    def pixelMetric(self, metric, option=None, widget=None):
        if metric == QStyle.PixelMetric.PM_SliderLength:
            return self._handle_size
        if metric == QStyle.PixelMetric.PM_SliderThickness:
            return max(self._handle_size + 6, self._groove_height + 10)
        return super().pixelMetric(metric, option, widget)

    def _groove_rect(self, option: QStyleOptionSlider) -> QRect:
        diameter = self._handle_size
        groove_h = self._groove_height
        inset = max(1, diameter // 2)
        width = max(2, int(option.rect.width()) - inset * 2)
        x = int(option.rect.left()) + inset
        y = int(option.rect.center().y() - groove_h // 2)
        return QRect(x, y, width, groove_h)

    def _handle_rect(self, option: QStyleOptionSlider) -> QRect:
        groove = self._groove_rect(option)
        diameter = self._handle_size
        available = max(0, groove.width() - diameter)
        pos = QStyle.sliderPositionFromValue(int(option.minimum), int(option.maximum), int(option.sliderPosition), int(available), bool(option.upsideDown))
        x = int(groove.left()) + int(pos)
        y = int(groove.center().y() - diameter // 2)
        return QRect(x, y, diameter, diameter)

    def subControlRect(self, control, option, sub_control, widget=None):
        if control == QStyle.ComplexControl.CC_Slider and isinstance(option, QStyleOptionSlider):
            if option.orientation == Qt.Horizontal:
                if sub_control == QStyle.SubControl.SC_SliderGroove:
                    return self._groove_rect(option)
                if sub_control == QStyle.SubControl.SC_SliderHandle:
                    return self._handle_rect(option)
        return super().subControlRect(control, option, sub_control, widget)

    def drawComplexControl(self, control, option, painter, widget=None):
        if control != QStyle.ComplexControl.CC_Slider or not isinstance(option, QStyleOptionSlider):
            super().drawComplexControl(control, option, painter, widget)
            return
        if option.orientation != Qt.Horizontal:
            super().drawComplexControl(control, option, painter, widget)
            return
        groove = self._groove_rect(option)
        handle = self._handle_rect(option)
        radius = max(2.0, groove.height() / 2.0)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        groove_color = QColor(self._groove_color)
        fill_color = QColor(self._fill_color)
        handle_color = QColor(self._handle_color)
        border_color = QColor(self._border_color)
        if not enabled:
            groove_color.setAlpha(125)
            fill_color = QColor(self._groove_color)
            fill_color = fill_color.lighter(112)
            fill_color.setAlpha(165)
            handle_color = QColor(self._border_color)
            handle_color = handle_color.lighter(128)
            handle_color.setAlpha(185)
            border_color.setAlpha(165)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(groove_color)
        painter.drawRoundedRect(QRectF(groove), radius, radius)
        if handle.isValid():
            if option.upsideDown:
                fill_left = float(handle.center().x())
                fill_width = float(groove.right() - handle.center().x())
            else:
                fill_left = float(groove.left())
                fill_width = float(handle.center().x() - groove.left())
            if fill_width > 0:
                fill_rect = QRectF(fill_left, float(groove.top()), fill_width, float(groove.height()))
                painter.setBrush(fill_color)
                painter.drawRoundedRect(fill_rect, radius, radius)
        if handle.isValid():
            circle_rect = QRectF(handle)
            painter.setBrush(handle_color)
            painter.setPen(QPen(border_color, max(1, int(round(self._handle_size / 16)))))
            painter.drawEllipse(circle_rect)
        painter.restore()

class SquareCheckBoxStyle(QProxyStyle):

    def __init__(self, *, border_color: str, fill_color: str, check_color: str, size: int=16, radius: int=4, parent: QWidget | None=None) -> None:
        super().__init__(parent.style() if parent is not None else None)
        self._border_color = QColor(border_color)
        self._fill_color = QColor(fill_color)
        self._check_color = QColor(check_color)
        self._size = max(12, int(size))
        self._radius = max(2, int(radius))

    def set_colors(self, *, border_color: str, fill_color: str, check_color: str) -> None:
        self._border_color = QColor(border_color)
        self._fill_color = QColor(fill_color)
        self._check_color = QColor(check_color)

    def set_metrics(self, *, size: int, radius: int) -> None:
        self._size = max(12, int(size))
        self._radius = max(2, int(radius))

    def pixelMetric(self, metric, option=None, widget=None):
        if metric in {QStyle.PixelMetric.PM_IndicatorWidth, QStyle.PixelMetric.PM_IndicatorHeight}:
            return self._size
        return super().pixelMetric(metric, option, widget)

    def _indicator_rect(self, option) -> QRect:
        left = int(option.rect.left()) + 2
        top = int(option.rect.center().y() - self._size // 2)
        return QRect(left, top, self._size, self._size)

    def subElementRect(self, element, option, widget=None):
        if isinstance(option, QStyleOptionButton):
            if element == QStyle.SubElement.SE_CheckBoxIndicator:
                return self._indicator_rect(option)
            if element == QStyle.SubElement.SE_CheckBoxContents:
                indicator = self._indicator_rect(option)
                gap = 6
                left = int(indicator.right() + 1 + gap)
                width = max(0, int(option.rect.right()) - left + 1)
                return QRect(left, int(option.rect.top()), width, int(option.rect.height()))
        return super().subElementRect(element, option, widget)

    def drawPrimitive(self, element, option, painter, widget=None):
        if element != QStyle.PrimitiveElement.PE_IndicatorCheckBox:
            super().drawPrimitive(element, option, painter, widget)
            return
        rect = option.rect.adjusted(1, 1, -2, -2)
        checked = bool(option.state & QStyle.StateFlag.State_On)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        border = QColor(self._border_color)
        fill = QColor(self._fill_color if checked else 'transparent')
        check = QColor(self._check_color)
        if not enabled:
            border.setAlpha(130)
            fill.setAlpha(110 if checked else 0)
            check.setAlpha(170)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(border, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(rect), float(self._radius), float(self._radius))
        if checked:
            pen_w = max(2, int(round(self._size / 9)))
            pen = QPen(check, pen_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            x = float(rect.x())
            y = float(rect.y())
            w = float(rect.width())
            h = float(rect.height())
            p1 = QPointF(x + w * 0.24, y + h * 0.56)
            p2 = QPointF(x + w * 0.44, y + h * 0.74)
            p3 = QPointF(x + w * 0.78, y + h * 0.34)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
        painter.restore()
