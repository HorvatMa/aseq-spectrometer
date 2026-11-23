"""Control of the ASEQ Instruments LR2 spectrometer (USB HID, VID 0xE220 / PID 0x0100).

Pure-Python reimplementation of ASEQ's ``spectrlib_shared.dll`` (protocol taken from
the DLL source in ``L:\\Drivers\\ASEQ spectrometer\\AllSoftware.zip``), so it works from
64-bit Python without the 32-bit DLL.  Only dependency: ``hidapi`` + ``numpy``.

Close the ASEQ program before using this - both would talk to the same device.

Python use::

    from aseq import ASEQSpectrometer
    with ASEQSpectrometer() as spec:
        spec.set_exposure(10)               # ms
        wl, counts = spec.measure(n_avg=5)  # nm, counts (3653 points, like the ASEQ app)

Command line / other programs::

    python aseq.py info
    python aseq.py measure -e 10 -n 5 -o spectrum.txt
    python aseq.py serve --port 5050        # line-based TCP server, see README.md

Exposure is limited to ``max_exposure_ms`` (default 1000 ms) because the detector is
not cooled; raise it explicitly if you really need longer exposures.
"""

from __future__ import annotations

import argparse
import functools
import socket
import struct
import sys
import threading
import time

import numpy as np

try:
    import hid
except ImportError as e:  # pragma: no cover
    raise ImportError("Install the 'hidapi' package: pip install hidapi") from e

VID, PID = 0xE220, 0x0100
DEFAULT_MAX_EXPOSURE_MS = 1000.0

# frame layout (full frame, no reduction): 32 leading dummy/shielded pixels,
# 3648 active pixels, 14 trailing dummy pixels = 3694.  The calibration and the ASEQ
# app use 3653 points starting at frame index 32.
FIRST_ACTIVE = 32
N_DARK_PIXELS = 32

_PIX_PER_PACKET = 30
_FLASH_PAYLOAD = 60
_FLASH_MAX_PACKETS = 100

_ERRORS = {
    500: "wrong id", 501: "device not found", 502: "connect failed",
    503: "device not initialised", 504: "write failed", 505: "read failed",
    506: "wrong answer", 507: "frame packet error", 508: "too many packets",
}


class SpectrometerError(RuntimeError):
    pass


def list_devices() -> list[str]:
    """Serial numbers of all connected ASEQ spectrometers."""
    return [d["serial_number"] for d in hid.enumerate(VID, PID)]


def _locked(fn):
    """Serialise device access (GUI, live view and TCP clients share one device)."""
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self.lock:
            return fn(self, *a, **kw)
    return wrapper


class ASEQSpectrometer:
    def __init__(self, serial: str | None = None,
                 max_exposure_ms: float = DEFAULT_MAX_EXPOSURE_MS,
                 timeout_ms: int = 500):
        self.lock = threading.RLock()
        #: callables(wavelengths, counts, info_dict) called after every measure()
        self.listeners: list = []
        self.max_exposure_ms = float(max_exposure_ms)
        self.timeout_ms = timeout_ms
        self._h = hid.device()
        try:
            if serial:
                self._h.open(VID, PID, serial)
            else:
                self._h.open(VID, PID)
        except OSError as e:
            raise SpectrometerError(
                f"Cannot open ASEQ spectrometer {serial or ''} (connected? ASEQ app closed?)"
            ) from e
        self.serial = self._h.get_serial_number_string()
        self._exposure_ticks: int | None = None
        #: last acquisition parameters sent (0x03) and when the CCD has settled
        self._acq_params: tuple | None = None
        self._settled_at = 0.0
        #: exposure times to wait after new acquisition parameters with the same
        #: exposure, and after a new exposure (see set_acquisition_parameters)
        self.settle_frames = 3
        self.settle_frames_new_exposure = 6
        self._n_pixels = self._set_frame_format(0, 3647, 0)
        self._wavelengths: np.ndarray | None = None
        self._cal_header: str | None = None
        self._norm: np.ndarray | None = None
        self._irr: np.ndarray | None = None
        self.set_external_trigger(0, 0)

    # ------------------------------------------------------------------ low level
    def _write(self, cmd: int, payload: bytes = b"") -> None:
        report = bytes([0, cmd]) + payload
        report += bytes(65 - len(report))
        if self._h.write(report) != 65:
            raise SpectrometerError(f"HID write failed (cmd 0x{cmd:02X})")

    def _read(self, expected: int, timeout_ms: int | None = None) -> bytes:
        r = bytes(self._h.read(64, timeout_ms or self.timeout_ms))
        if len(r) != 64:
            raise SpectrometerError(f"HID read timeout (expected reply 0x{expected:02X})")
        if r[0] != expected:
            raise SpectrometerError(f"Unexpected reply 0x{r[0]:02X}, expected 0x{expected:02X}")
        return r

    def _cmd(self, cmd: int, payload: bytes = b"", check_error: bool = True) -> bytes:
        self._write(cmd, payload)
        r = self._read(cmd | 0x80)
        if check_error and r[1] != 0:
            raise SpectrometerError(f"Device error {r[1]} on command 0x{cmd:02X}")
        return r

    # ------------------------------------------------------------------ device commands
    @_locked
    def _set_frame_format(self, start: int, end: int, reduction: int) -> int:
        r = self._cmd(0x04, struct.pack("<HHB", start, end, reduction))
        return r[2] | r[3] << 8

    @_locked
    def get_frame_format(self) -> tuple[int, int, int, int]:
        """(start element, end element, reduction mode, pixels in frame)"""
        r = self._cmd(0x08, check_error=False)
        return struct.unpack("<HHBH", r[1:8])

    @_locked
    def set_acquisition_parameters(self, n_scans: int, n_blank: int, scan_mode: int,
                                   exposure_ticks: int) -> None:
        """Send acquisition parameters (0x03), only if they changed.

        Every 0x03 restarts the CCD integration: the frames that follow are too low
        for about two exposure times (tested on Linux 23. 11. 2026 at 1000 ms, first
        frame of each call 0.30-0.64 of the steady value, 0.83-0.97 for the next two).
        So the command is skipped when nothing changed, and after a change the next
        trigger waits ``settle_frames`` exposure times (see ``_wait_settled``).
        A new exposure time needs longer: after 100 -> 1000 ms with a 3-exposure wait
        the first frame was still 0.91 (ramping to 0.98); with 6 it was 0.98-0.99, within
        the noise. Hence ``settle_frames_new_exposure``.
        """
        self._check_ticks(exposure_ticks)
        params = (n_scans, n_blank, scan_mode, exposure_ticks)
        if params == self._acq_params:
            return
        new_exposure = self._acq_params is None or self._acq_params[3] != exposure_ticks
        self._cmd(0x03, struct.pack("<HHBI", *params))
        self._acq_params = params
        self._exposure_ticks = exposure_ticks
        k = self.settle_frames_new_exposure if new_exposure else self.settle_frames
        self._settled_at = time.monotonic() + k * exposure_ticks * 1e-5 + 0.005

    def _wait_settled(self) -> None:
        dt = self._settled_at - time.monotonic()
        if dt > 0:
            time.sleep(dt)

    @_locked
    def get_acquisition_parameters(self) -> tuple[int, int, int, int]:
        """(scans, blank scans, scan mode, exposure in 10 us ticks)"""
        r = self._cmd(0x09, check_error=False)
        return struct.unpack("<HHBI", r[1:10])

    @_locked
    def set_external_trigger(self, enable: int, edge: int) -> None:
        self._cmd(0x05, bytes([enable, edge]))

    @_locked
    def get_status(self) -> tuple[int, int]:
        """(flags, frames in memory); flags & 1 = busy, flags & 2 = memory full"""
        r = self._cmd(0x01, check_error=False)
        return r[1], r[2] | r[3] << 8

    @_locked
    def clear_memory(self) -> None:
        self._cmd(0x07)

    @_locked
    def trigger(self) -> None:
        self._write(0x06)

    @_locked
    def get_frame(self, index: int = 0) -> np.ndarray:
        """Raw frame (uint16, ``n_pixels`` long) from device memory."""
        n_packets = -(-self._n_pixels // _PIX_PER_PACKET)
        self._write(0x0A, struct.pack("<HHB", 0, index, n_packets))
        buf = np.zeros(n_packets * _PIX_PER_PACKET, np.uint16)
        for i in range(n_packets):
            r = self._read(0x8A)
            left = r[3]
            if left >= 250 or left != n_packets - i - 1:
                raise SpectrometerError(f"Frame transfer error (remaining={left})")
            off = r[1] | r[2] << 8
            buf[off:off + _PIX_PER_PACKET] = np.frombuffer(r[4:64], "<u2")
        return buf[:self._n_pixels]

    @_locked
    def read_flash(self, offset: int, n_bytes: int) -> bytes:
        out = bytearray()
        while len(out) < n_bytes:
            want = min(_FLASH_MAX_PACKETS, -(-(n_bytes - len(out)) // _FLASH_PAYLOAD))
            self._write(0x1A, struct.pack("<IB", offset + len(out), want))
            chunk = bytearray(want * _FLASH_PAYLOAD)
            for i in range(want):
                r = self._read(0x9A)
                if r[3] >= 250:
                    raise SpectrometerError("Flash read error")
                lo = r[1] | r[2] << 8
                chunk[lo:lo + _FLASH_PAYLOAD] = r[4:64]
            out += chunk
        return bytes(out[:n_bytes])

    # ------------------------------------------------------------------ calibration
    @_locked
    def _load_calibration(self) -> None:
        raw = bytearray()
        while len(raw) < 0x20000:  # 128 kB user flash, calibration is text ~96 kB
            block = self.read_flash(len(raw), 6000)
            raw += block
            if 0xFF in block:
                break
        text = bytes(raw).split(b"\xff", 1)[0].decode("ascii", "replace")
        lines = [ln.strip() for ln in text.splitlines()]
        self._cal_header = lines[0]

        def block(start: int) -> np.ndarray:
            vals = []
            for ln in lines[start:]:
                if not ln:
                    break
                vals.append(float(ln))
            return np.asarray(vals)

        self._wavelengths = block(12)
        n = len(self._wavelengths)
        self._norm = block(13 + n)
        self._irr = block(14 + 2 * n)

    @property
    def wavelengths(self) -> np.ndarray:
        """Wavelength (nm) of each returned spectrum point (from device flash)."""
        if self._wavelengths is None:
            self._load_calibration()
        return self._wavelengths

    @property
    def calibration_header(self) -> str:
        if self._cal_header is None:
            self._load_calibration()
        return self._cal_header

    @property
    def intensity_normalization(self) -> np.ndarray:
        if self._norm is None:
            self._load_calibration()
        return self._norm

    # ------------------------------------------------------------------ high level
    def _check_ticks(self, ticks: int) -> None:
        if ticks < 1:
            raise ValueError("Exposure must be at least 0.01 ms")
        if ticks * 0.01 > self.max_exposure_ms + 1e-9:
            raise ValueError(
                f"Exposure {ticks * 0.01:g} ms exceeds the safety limit "
                f"{self.max_exposure_ms:g} ms (detector is not cooled). "
                f"Pass max_exposure_ms=... to raise it.")

    @_locked
    def set_exposure(self, exposure_ms: float) -> float:
        """Set integration time in ms (resolution 0.01 ms). Returns the value set."""
        ticks = int(round(exposure_ms * 100))
        self._check_ticks(ticks)
        # continuous mode (0): the CCD is read all the time and frames are stored on
        # trigger.  The idle modes (1, 2) give an over-exposed first frame.
        n = self._acq_params[0] if self._acq_params else 1   # keep n: no second restart
        self.set_acquisition_parameters(n, 0, 0, ticks)
        self._wait_settled()  # let the CCD settle at the new exposure
        return ticks / 100

    @property
    def exposure_ms(self) -> float:
        if self._exposure_ticks is None:
            self._exposure_ticks = self.get_acquisition_parameters()[3]
        return self._exposure_ticks / 100

    @_locked
    def acquire_frames(self, n: int = 1, timeout_s: float | None = None) -> np.ndarray:
        """Take ``n`` consecutive raw frames (shape (n, n_pixels), uint16)."""
        if not 1 <= n <= 137:
            raise ValueError("n must be 1..137 (device memory)")
        ticks = self._exposure_ticks or self.get_acquisition_parameters()[3]
        self._check_ticks(ticks)
        self.set_acquisition_parameters(n, 0, 0, ticks)   # no-op if unchanged
        self._wait_settled()
        self.clear_memory()
        self.trigger()
        exp_s = ticks * 1e-5
        deadline = time.monotonic() + (timeout_s or (n + 2) * exp_s + 2.0)
        while True:
            flags, n_mem = self.get_status()
            if n_mem >= n and not flags & 1:
                break
            if time.monotonic() > deadline:
                raise SpectrometerError(f"Acquisition timeout (flags={flags}, frames={n_mem})")
            time.sleep(min(0.01, exp_s / 4))
        return np.stack([self.get_frame(i) for i in range(n)])

    @_locked
    def measure(self, exposure_ms: float | None = None, n_avg: int = 1,
                subtract_dark: bool = False, normalize: bool = False
                ) -> tuple[np.ndarray, np.ndarray]:
        """Take a spectrum.

        Returns (wavelengths [nm], counts) with the same 3653 points as the ASEQ app.
        ``n_avg`` frames are averaged.  ``subtract_dark`` subtracts the mean of the
        leading shielded pixels (electronic offset, ~2900 counts); ``normalize``
        applies the intensity normalisation coefficients stored in the device.
        """
        if exposure_ms is not None and (self._exposure_ticks is None
                                        or round(exposure_ms * 100) != self._exposure_ticks):
            self.set_exposure(exposure_ms)
        frames = self.acquire_frames(n_avg).astype(float)
        wl = self.wavelengths
        spec = frames[:, FIRST_ACTIVE:FIRST_ACTIVE + len(wl)].mean(axis=0)
        if subtract_dark:
            spec -= frames[:, 4:N_DARK_PIXELS].mean()
        if normalize and len(self.intensity_normalization) == len(spec):
            spec *= self.intensity_normalization
        info = dict(exposure_ms=self.exposure_ms, n_avg=n_avg, dark=subtract_dark,
                    norm=normalize, time=time.time(),
                    source=threading.current_thread().name)
        for cb in list(self.listeners):
            try:
                cb(wl, spec, info)
            except Exception:
                pass
        return wl, spec

    @_locked
    def close(self) -> None:
        if self._h is not None:
            self._h.close()
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- file output
def save_spectrum(path: str, wl: np.ndarray, counts: np.ndarray, header: str = "") -> None:
    """Two-column text file (wavelength, counts) - same layout as the ASEQ app."""
    np.savetxt(path, np.column_stack([wl, counts]), fmt=["%.3f", "%.3f"],
               delimiter="\t", header=header)


# ---------------------------------------------------------------------- TCP server
_HELP = ("commands: EXP <ms> | EXP? | MEAS [n_avg] [DARK] [NORM] | WL? | SAVE <file> "
         "[n_avg] [DARK] [NORM] | INFO? | QUIT")


def _handle(spec: ASEQSpectrometer, line: str) -> str:
    parts = line.split()
    if not parts:
        return ""
    cmd, args = parts[0].upper(), parts[1:]
    flags = {a.upper() for a in args}
    nums = [a for a in args if a.replace(".", "", 1).isdigit()]
    if cmd == "EXP":
        return f"OK {spec.set_exposure(float(args[0])):g}"
    if cmd == "EXP?":
        return f"{spec.exposure_ms:g}"
    if cmd == "WL?":
        return " ".join(f"{w:.3f}" for w in spec.wavelengths)
    if cmd == "INFO?":
        return f"{spec.serial} | {spec.calibration_header} | exposure {spec.exposure_ms:g} ms"
    if cmd in ("MEAS", "SAVE"):
        if cmd == "SAVE":
            path, nums = args[0], [a for a in args[1:] if a.isdigit()]
        n = int(nums[0]) if nums else 1
        wl, c = spec.measure(n_avg=n, subtract_dark="DARK" in flags, normalize="NORM" in flags)
        if cmd == "SAVE":
            save_spectrum(path, wl, c, f"exposure_ms={spec.exposure_ms:g} n_avg={n}")
            return f"OK {path}"
        return " ".join(f"{v:.3f}" for v in c)
    if cmd in ("HELP", "?"):
        return _HELP
    return f"ERR unknown command; {_HELP}"


class CommandServer:
    """Line-based TCP server (one command per line, one reply line per command).

    Runs in background threads; ``on_log(client, command, reply)`` is called for every
    command (and with command=None on connect/disconnect).
    """

    def __init__(self, spec: ASEQSpectrometer, host: str = "127.0.0.1", port: int = 5050,
                 on_log=None):
        self.spec, self.host, self.port = spec, host, port
        self.on_log = on_log
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()

    @property
    def running(self) -> bool:
        return self._sock is not None

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    def _log(self, client, command, reply):
        if self.on_log:
            try:
                self.on_log(client, command, reply)
            except Exception:
                pass

    def start(self) -> None:
        self._sock = socket.create_server((self.host, self.port))
        self._sock.settimeout(0.3)
        self._thread = threading.Thread(target=self._accept_loop, name="TCP server",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        sock, self._sock = self._sock, None
        if sock:
            sock.close()
        for c in list(self._clients):
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread:
            self._thread.join(2)

    def _accept_loop(self) -> None:
        while self._sock is not None:
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except (OSError, AttributeError):
                break
            name = f"{addr[0]}:{addr[1]}"
            threading.Thread(target=self._client, args=(conn, name), name=f"TCP {name}",
                             daemon=True).start()

    def _client(self, conn: socket.socket, name: str) -> None:
        self._clients.add(conn)
        self._log(name, None, "connected")
        try:
            with conn, conn.makefile("rw", encoding="ascii", errors="replace",
                                     newline="\n") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.upper() == "QUIT":
                        break
                    try:
                        reply = _handle(self.spec, line)
                    except Exception as e:  # report errors to the client, keep serving
                        reply = f"ERR {e}"
                    self._log(name, line, reply)
                    f.write(reply + "\n")
                    f.flush()
        except (OSError, ValueError):
            pass
        finally:
            self._clients.discard(conn)
            self._log(name, None, "disconnected")


def short_reply(reply: str, n: int = 80) -> str:
    """Shorten long replies (spectra) for logging."""
    if len(reply) <= n or reply.startswith(("ERR", "OK")) or not reply[:1].lstrip("-").isdigit():
        return reply
    return f"{reply[:n - 20]} ... ({len(reply.split())} values)"


def serve(spec: ASEQSpectrometer, host: str = "127.0.0.1", port: int = 5050) -> None:
    """Run the command server in the console until Ctrl+C."""
    def log(client, cmd, reply):
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {client}  {reply}" if cmd is None
              else f"{stamp} {client}  {cmd}  ->  {short_reply(reply)}", flush=True)

    srv = CommandServer(spec, host, port, log)
    srv.start()
    print(f"ASEQ {spec.serial} listening on {host}:{port}  ({_HELP})", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


# ---------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ASEQ spectrometer control")
    p.add_argument("--serial", help="device serial (default: first found)")
    p.add_argument("--max-exposure", type=float, default=DEFAULT_MAX_EXPOSURE_MS,
                   help="safety limit for exposure in ms (default %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list connected spectrometers")
    sub.add_parser("info", help="device info and current settings")
    m = sub.add_parser("measure", help="take one (averaged) spectrum")
    m.add_argument("-e", "--exposure", type=float, required=True, help="integration time in ms")
    m.add_argument("-n", "--avg", type=int, default=1, help="frames to average")
    m.add_argument("-o", "--out", help="output file (default: print to stdout)")
    m.add_argument("--dark", action="store_true", help="subtract electronic offset")
    m.add_argument("--norm", action="store_true", help="apply intensity normalisation")
    s = sub.add_parser("serve", help="run a TCP command server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5050)
    a = p.parse_args(argv)

    if a.cmd == "list":
        print("\n".join(list_devices()) or "no ASEQ spectrometer found")
        return 0
    with ASEQSpectrometer(a.serial, max_exposure_ms=a.max_exposure) as spec:
        if a.cmd == "info":
            wl = spec.wavelengths
            print(f"serial:      {spec.serial}")
            print(f"calibration: {spec.calibration_header}")
            print(f"range:       {wl[0]:.2f} - {wl[-1]:.2f} nm, {len(wl)} points")
            print(f"frame:       {spec.get_frame_format()}")
            print(f"acquisition: {spec.get_acquisition_parameters()} (scans, blank, mode, 10us ticks)")
        elif a.cmd == "measure":
            wl, c = spec.measure(a.exposure, a.avg, a.dark, a.norm)
            hdr = f"ASEQ {spec.serial} exposure_ms={spec.exposure_ms:g} n_avg={a.avg}"
            save_spectrum(a.out or sys.stdout, wl, c, hdr)
        elif a.cmd == "serve":
            serve(spec, a.host, a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
