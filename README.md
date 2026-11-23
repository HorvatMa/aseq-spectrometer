# ASEQ spectrometer control

`aseq.py` drives the ASEQ LR2 spectrometer (serial `ASQ_SPC2359373`, calibration
`LR2B4.0 c.Y 1383`, 294.9–999.4 nm, 3653 points) directly over USB-HID: a pure-Python
port of ASEQ's `spectrlib_shared.dll`. No ASEQ software or DLL needed.

**Close the ASEQ program first**: both would send commands to the same device.

**Exposure limit:** at most 1000 ms by default, because the detector is not cooled.
Longer exposures raise an error before anything goes to the device. To raise the limit,
pass `max_exposure_ms=...` or `--max-exposure ...` on purpose.

## Install
```
pip install -r requirements.txt        # hidapi, numpy, PyQt5, pyqtgraph
```
or without installing anything: `uv run --no-project --with hidapi --with numpy --with PyQt5 --with pyqtgraph python gui.py`

## GUI
```
python gui.py                 # connects, starts the TCP server on 127.0.0.1:5050
python gui.py --port 5051 --no-server --host 0.0.0.0
```
- The plot shows the latest spectrum, whatever started it: *Measure*, *Live*, or a TCP
  client. The title line gives the time, exposure, averaging and source (`GUI`, `live`,
  or `TCP <client address>`).
- The command log lists every TCP command with its reply (spectra are shortened, errors
  are red), plus client connects/disconnects and GUI actions. *Save log…* writes it to a file.
- The exposure box (max 1000 ms) follows changes made over TCP. *Live* measures
  continuously and leaves gaps so TCP commands still get through.
- The GUI and its TCP server share one connection to the device, so use the GUI's server
  instead of running `aseq.py serve` at the same time.

## Python
```python
from aseq import ASEQSpectrometer
with ASEQSpectrometer() as spec:
    spec.set_exposure(10)                    # ms, 0.01 ms steps
    wl, counts = spec.measure(n_avg=5, subtract_dark=True)
    wl, counts = spec.measure(exposure_ms=50)  # change exposure and measure
```
- `measure()` returns wavelengths (nm, from the device's flash calibration) and counts
  on the same 3653 points as the ASEQ app's text export.
- `subtract_dark=True` subtracts the mean of the light-shielded pixels in the same
  frames. This is the electronic offset, which drifts (seen anywhere from ~1400 to ~2900 counts).
  It is not a dark spectrum: take one with the light blocked for that.
- `normalize=True` applies the intensity-normalisation coefficients stored in the device.
- `acquire_frames(n)` gives raw 3694-pixel frames (32 shielded/dummy + 3648 active + 14 dummy).
- Each frame takes about 0.13 s to read over USB, on top of the exposure time. Reading the
  calibration on the first `measure()` takes about 2 s.

## Command line (callable from any program)
```
python aseq.py list
python aseq.py info
python aseq.py measure -e 10 -n 5 --dark -o spectrum.txt   # wavelength<TAB>counts
```

## TCP server (keeps the device open; for LabVIEW, Mathematica, other processes…)
```
python aseq.py serve --port 5050
```
Send one command per line and read one reply line back:

| command | reply |
|---|---|
| `EXP 10` | `OK 10`: integration time in ms |
| `EXP?` | current exposure in ms |
| `MEAS [n_avg] [DARK] [NORM]` | 3653 space-separated counts |
| `WL?` | 3653 wavelengths (nm) |
| `SAVE <file> [n_avg] [DARK] [NORM]` | `OK <file>` |
| `INFO?` | serial, calibration, exposure |
| `QUIT` | closes the connection |

Errors come back as `ERR <message>`.

Python client example:
```python
import socket
f = socket.create_connection(("127.0.0.1", 5050)).makefile("rw", newline="\n")
def ask(cmd): f.write(cmd + "\n"); f.flush(); return f.readline().strip()
ask("EXP 20"); counts = list(map(float, ask("MEAS 3 DARK").split()))
```

## Notes
- The driver uses continuous scan mode. In the "idle" modes, the first frame after a pause
  carries leftover charge (it was saturated near 17k counts in tests).
- The pixel→wavelength mapping (frame index 32 = first calibration point) comes from
  the CCD layout and was not checked against a line lamp. Check it with a known line
  (e.g. Hg 546.07 nm) before relying on absolute wavelengths.
