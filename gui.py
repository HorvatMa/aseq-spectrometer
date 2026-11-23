"""PyQt5 GUI for the ASEQ spectrometer.

Shows the most recent spectrum (whoever triggered it: this GUI, live view or a TCP
client) and logs every command received over TCP.

    python gui.py                    # connect to the first spectrometer, start server on 5050
    python gui.py --port 5051 --no-server --max-exposure 1000
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

import aseq


class Bridge(QtCore.QObject):
    """Signals emitted from worker / server threads, delivered in the GUI thread."""
    log = QtCore.pyqtSignal(str, str)                  # text, colour
    spectrum = QtCore.pyqtSignal(object, object, dict)  # wavelengths, counts, info
    busy = QtCore.pyqtSignal(bool)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("ASEQ spectrometer")
        self.resize(1200, 750)
        self.args = args
        self.spec: aseq.ASEQSpectrometer | None = None
        self.server: aseq.CommandServer | None = None
        self.last: tuple | None = None
        self._live = False
        self._live_thread: threading.Thread | None = None
        self._acq = dict(n_avg=1, dark=False, norm=False)

        self.bridge = Bridge()
        self.bridge.log.connect(self._append_log)
        self.bridge.spectrum.connect(self._show_spectrum)

        self._build_ui()
        self._refresh_devices()

        self._timer = QtCore.QTimer(self, interval=300, timeout=self._poll)
        self._timer.start()

        if self.devices.count():
            self._connect()
            if self.spec and args.server:
                self._toggle_server()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "wavelength", "nm")
        self.plot.setLabel("left", "counts")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#1f77b4", width=1))
        self.plot_title = QtWidgets.QLabel("no spectrum yet")
        self.cursor_label = QtWidgets.QLabel("")
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

        plot_box = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(plot_box)
        pl.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.plot_title, 1)
        row.addWidget(self.cursor_label)
        pl.addLayout(row)
        pl.addWidget(self.plot)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        log_box = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(log_box)
        ll.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>Command log</b>"), 1)
        b = QtWidgets.QPushButton("Save log…", clicked=self._save_log)
        row.addWidget(b)
        row.addWidget(QtWidgets.QPushButton("Clear", clicked=self.log.clear))
        ll.addLayout(row)
        ll.addWidget(self.log)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(plot_box)
        split.addWidget(log_box)
        split.setSizes([500, 250])

        # --- side panel
        side = QtWidgets.QWidget()
        side.setFixedWidth(280)
        sl = QtWidgets.QVBoxLayout(side)

        g = QtWidgets.QGroupBox("Device")
        f = QtWidgets.QFormLayout(g)
        self.devices = QtWidgets.QComboBox()
        rb = QtWidgets.QPushButton("⟳", clicked=self._refresh_devices, maximumWidth=30)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.devices, 1)
        row.addWidget(rb)
        f.addRow(row)
        self.connect_btn = QtWidgets.QPushButton("Connect", clicked=self._toggle_connect)
        f.addRow(self.connect_btn)
        self.dev_info = QtWidgets.QLabel("not connected", wordWrap=True)
        f.addRow(self.dev_info)
        sl.addWidget(g)

        g = QtWidgets.QGroupBox("Acquisition")
        f = QtWidgets.QFormLayout(g)
        self.exposure = QtWidgets.QDoubleSpinBox(decimals=2, suffix=" ms",
                                                 keyboardTracking=False)
        self.exposure.setRange(0.01, self.args.max_exposure)
        self.exposure.setValue(10)
        self.exposure.setToolTip(f"Integration time, max {self.args.max_exposure:g} ms "
                                 "(detector not cooled)")
        self.exposure.valueChanged.connect(self._set_exposure)
        f.addRow("Exposure", self.exposure)
        self.n_avg = QtWidgets.QSpinBox(minimum=1, maximum=137, value=1)
        self.n_avg.valueChanged.connect(lambda v: self._acq.update(n_avg=v))
        f.addRow("Average", self.n_avg)
        self.dark = QtWidgets.QCheckBox("Subtract offset (shielded pixels)")
        self.dark.toggled.connect(lambda v: self._acq.update(dark=v))
        f.addRow(self.dark)
        self.norm = QtWidgets.QCheckBox("Intensity normalisation")
        self.norm.toggled.connect(lambda v: self._acq.update(norm=v))
        f.addRow(self.norm)
        self.measure_btn = QtWidgets.QPushButton("Measure", clicked=self._measure_once)
        self.live_btn = QtWidgets.QPushButton("Live", checkable=True, toggled=self._toggle_live)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.measure_btn)
        row.addWidget(self.live_btn)
        f.addRow(row)
        f.addRow(QtWidgets.QPushButton("Save spectrum…", clicked=self._save_spectrum))
        sl.addWidget(g)

        g = QtWidgets.QGroupBox("TCP command server")
        f = QtWidgets.QFormLayout(g)
        self.host = QtWidgets.QLineEdit(self.args.host)
        f.addRow("Host", self.host)
        self.port = QtWidgets.QSpinBox(minimum=1, maximum=65535, value=self.args.port)
        f.addRow("Port", self.port)
        self.server_btn = QtWidgets.QPushButton("Start server", clicked=self._toggle_server)
        f.addRow(self.server_btn)
        self.server_info = QtWidgets.QLabel("stopped")
        f.addRow(self.server_info)
        sl.addWidget(g)
        sl.addStretch(1)

        central = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(central)
        cl.addWidget(split, 1)
        cl.addWidget(side)
        self.setCentralWidget(central)
        self._update_enabled()

    def _update_enabled(self):
        on = self.spec is not None
        for w in (self.exposure, self.n_avg, self.dark, self.norm, self.measure_btn,
                  self.live_btn, self.server_btn):
            w.setEnabled(on)
        self.devices.setEnabled(not on)
        self.connect_btn.setText("Disconnect" if on else "Connect")

    # ------------------------------------------------------------------ helpers
    def log_msg(self, text: str, colour: str = "") -> None:
        """Thread-safe logging."""
        self.bridge.log.emit(text, colour)

    def _append_log(self, text: str, colour: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"{stamp}  {text}".replace("&", "&amp;").replace("<", "&lt;")
        if colour:
            line = f'<span style="color:{colour}">{line}</span>'
        self.log.appendHtml(line)

    def _bg(self, fn, name="GUI"):
        """Run device work in a background thread, log exceptions."""
        def run():
            try:
                fn()
            except Exception as e:
                self.log_msg(f"[GUI] error: {e}", "red")
        threading.Thread(target=run, name=name, daemon=True).start()

    # ------------------------------------------------------------------ device
    def _refresh_devices(self):
        if self.spec:
            return
        self.devices.clear()
        self.devices.addItems(aseq.list_devices())

    def _toggle_connect(self):
        if self.spec:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        try:
            spec = aseq.ASEQSpectrometer(self.devices.currentText() or None,
                                         max_exposure_ms=self.args.max_exposure)
            wl = spec.wavelengths
        except Exception as e:
            self.log_msg(f"[GUI] cannot connect: {e}", "red")
            return
        self.spec = spec
        spec.listeners.append(lambda w, c, i: self.bridge.spectrum.emit(w, c, i))
        cur = spec.exposure_ms
        if cur > self.args.max_exposure:  # left long by another program: clamp
            cur = spec.set_exposure(self.args.max_exposure)
        self.exposure.blockSignals(True)
        self.exposure.setValue(cur)
        self.exposure.blockSignals(False)
        self.dev_info.setText(f"{spec.serial}\n{spec.calibration_header}\n"
                              f"{wl[0]:.1f}–{wl[-1]:.1f} nm, {len(wl)} pts")
        self.log_msg(f"[GUI] connected {spec.serial}, exposure {cur:g} ms", "green")
        self._update_enabled()

    def _disconnect(self):
        self.live_btn.setChecked(False)
        self._stop_live()
        if self.server:
            self._toggle_server()
        if self.spec:
            self.spec.close()
            self.log_msg(f"[GUI] disconnected {self.spec.serial}")
        self.spec = None
        self._update_enabled()

    # ------------------------------------------------------------------ acquisition
    def _set_exposure(self, value):
        spec = self.spec
        if spec:
            self._bg(lambda: self.log_msg(f"[GUI] exposure {spec.set_exposure(value):g} ms"))

    def _measure_kwargs(self):
        a = self._acq
        return dict(n_avg=a["n_avg"], subtract_dark=a["dark"], normalize=a["norm"])

    def _measure_once(self):
        spec = self.spec
        if spec:
            self._bg(lambda: spec.measure(**self._measure_kwargs()))

    def _toggle_live(self, on):
        if on:
            self._live = True
            self._live_thread = threading.Thread(target=self._live_loop, name="live",
                                                 daemon=True)
            self._live_thread.start()
            self.log_msg("[GUI] live view started")
        else:
            self._stop_live()

    def _stop_live(self):
        if self._live:
            self._live = False
            if self._live_thread:
                self._live_thread.join(5)
            self.log_msg("[GUI] live view stopped")

    def _live_loop(self):
        while self._live and self.spec:
            try:
                self.spec.measure(**self._measure_kwargs())
            except Exception as e:
                self.log_msg(f"[live] error: {e}", "red")
                time.sleep(1)
            time.sleep(0.03)  # let TCP clients get the device in between

    def _show_spectrum(self, wl, counts, info):
        self.last = (wl, counts, info)
        self.curve.setData(wl, counts)
        src = info["source"]
        src = "GUI" if src in ("GUI", "MainThread") else src
        self.plot_title.setText(
            f"{time.strftime('%H:%M:%S', time.localtime(info['time']))}   "
            f"exposure {info['exposure_ms']:g} ms   avg {info['n_avg']}"
            f"{'   offset subtracted' if info['dark'] else ''}"
            f"{'   normalised' if info['norm'] else ''}   "
            f"max {np.max(counts):.0f}   from: {src}")

    def _mouse_moved(self, pos):
        if self.plot.sceneBoundingRect().contains(pos):
            p = self.plot.getPlotItem().vb.mapSceneToView(pos)
            self.cursor_label.setText(f"{p.x():.2f} nm, {p.y():.1f}")

    def _save_spectrum(self):
        if not self.last:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save spectrum", time.strftime("spectrum_%Y%m%d_%H%M%S.txt"),
            "Text (*.txt);;All files (*)")
        if path:
            wl, c, i = self.last
            aseq.save_spectrum(path, wl, c, f"ASEQ {self.spec.serial if self.spec else ''} "
                               f"exposure_ms={i['exposure_ms']:g} n_avg={i['n_avg']} "
                               f"dark={i['dark']} norm={i['norm']}")
            self.log_msg(f"[GUI] saved {path}")

    def _save_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save log", time.strftime("aseq_log_%Y%m%d_%H%M%S.txt"), "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.log.toPlainText())

    # ------------------------------------------------------------------ server
    def _toggle_server(self):
        if self.server:
            self.server.stop()
            self.server = None
            self.log_msg("[server] stopped")
        elif self.spec:
            srv = aseq.CommandServer(self.spec, self.host.text(), self.port.value(),
                                     self._server_log)
            try:
                srv.start()
            except OSError as e:
                self.log_msg(f"[server] cannot start on {self.host.text()}:"
                             f"{self.port.value()}: {e}", "red")
                return
            self.server = srv
            self.log_msg(f"[server] listening on {srv.host}:{srv.port}", "green")
        running = self.server is not None
        self.server_btn.setText("Stop server" if running else "Start server")
        self.host.setEnabled(not running)
        self.port.setEnabled(not running)

    def _server_log(self, client, command, reply):
        if command is None:
            self.log_msg(f"[{client}] {reply}", "gray")
        else:
            colour = "red" if reply.startswith("ERR") else "blue"
            self.log_msg(f"[{client}] {command}  ->  {aseq.short_reply(reply)}", colour)

    # ------------------------------------------------------------------ misc
    def _poll(self):
        if self.server:
            self.server_info.setText(f"running, {self.server.n_clients} client(s)")
        else:
            self.server_info.setText("stopped")
        # follow exposure changes made over TCP
        if self.spec and not self.exposure.hasFocus() and self.spec._exposure_ticks:
            cur = self.spec._exposure_ticks / 100
            if abs(cur - self.exposure.value()) > 1e-9:
                self.exposure.blockSignals(True)
                self.exposure.setValue(cur)
                self.exposure.blockSignals(False)

    def closeEvent(self, ev):
        self._disconnect()
        super().closeEvent(ev)


def main(argv=None):
    p = argparse.ArgumentParser(description="ASEQ spectrometer GUI")
    p.add_argument("--host", default="127.0.0.1",
                   help="server address (0.0.0.0 to accept other PCs)")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--no-server", dest="server", action="store_false",
                   help="don't start the TCP server automatically")
    p.add_argument("--max-exposure", type=float, default=aseq.DEFAULT_MAX_EXPOSURE_MS,
                   help="exposure safety limit in ms (default %(default)s)")
    args = p.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(args)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
