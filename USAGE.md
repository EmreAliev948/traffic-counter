# Traffic Counter – User Guide

This guide explains **what each part of the project does** and **which commands to run**.
It does not go into how anything works inside.

All commands are run from the `v3` folder in a terminal.

---

## 1. What the project does

The program watches live public traffic cameras. It recognises cars, trucks, buses,
motorcycles, bicycles and pedestrians, and counts every time one of them crosses a line
you draw on the picture. For each crossing it also records the direction ("in" or "out").

- Only the counts are saved (time, camera, type of object, direction).
- **No video and no pictures are ever saved.**
- You can export charts and tables for your report.

---

## 2. One-time setup

You need **Python 3.11, 3.12 or 3.13**. Newer versions (3.14, 3.15) do not work yet,
because the AI library (PyTorch) is not available for them.

Always run these commands from the **`v3` folder** (not from `tools/` or `traffic/`):

```bash
cd v3
python3.13 -m venv .venv             # use python3.13, not plain python3
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python --version                     # must show 3.11, 3.12 or 3.13
pip install -r requirements.txt
```

If `v3/.venv` already exists, skip the `venv` line and just activate it.

Each time you open a new terminal, activate the environment again first:

```bash
cd v3
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

The first time the program runs, it downloads the AI model (about 5 MB) by itself.

---

## 3. Overview of all tools

| Tool | What it is for | Command |
|---|---|---|
| **Counter** | Watches the cameras and counts traffic. Leave it running. | `python -m traffic run` |
| **Calibrate** | Draw the counting line for a camera with two mouse clicks | `python -m traffic calibrate <camera>` |
| **Manual counter** | Count traffic yourself by pressing keys (used to check accuracy) | `python -m traffic manual-count <camera>` |
| **Evaluate** | Compare your manual count with the automatic count | `python -m traffic evaluate <file.csv>` |
| **Report** | Save charts (pictures) and tables for the written report | `python -m traffic report` |

Every tool can also be started as `python -m tools.<name>` (for example `python -m tools.calibrate 2`).

---

## 4. Cameras (`config.yaml`)

All cameras are listed in the file `config.yaml`. Each camera has:

- `id` – a short number or name you use in commands (for example `"2"`)
- `name` – a readable description
- `url` – the stream address: a direct `.m3u8` or `rtsp://` link (web pages such as YouTube are not supported)
- `enabled` – `true` = counted when you run the counter, `false` = skipped
- `line` – the counting line (filled in by the Calibrate tool)

### Adding a new camera

1. Add a new entry at the end of `config.yaml`:

   ```yaml
     - id: "20"
       name: Sofia - example street
       url: http://example.com/cam/index.m3u8
       enabled: true
       line: [[0, 0], [1, 1]]     # temporary, replaced in step 2
   ```

2. Draw the line with the Calibrate tool (next section).

---

## 5. Calibrate – draw the counting line

```bash
python -m traffic calibrate 20
```

One picture from the camera opens on screen.

1. Click the **start** of the line, then the **end** of the line, across the road.
2. An arrow shows which direction will be counted as **"in"**. The opposite direction is **"out"**.
3. Press **Enter** to accept. Press **r** to start over, or **q** to cancel.

The program prints a line like:

```
    line: [[120, 400], [1100, 400]]
```

Copy it into `config.yaml` under that camera, replacing the old `line:`.

---

## 6. Counter – count traffic

```bash
python -m traffic run                      # all cameras with enabled: true
python -m traffic run --only 2,5           # only cameras 2 and 5 (even if disabled)
python -m traffic run --only 2 --preview   # also show a live window with boxes and the line
```

- **Stop it** with `Ctrl+C`. Everything counted so far is saved.
- In preview mode, press **q** in the window to stop.
- If a camera stream drops, the program reconnects by itself. You don't need to do anything.
- Leave it running for as long as you want data: **at least 7 days** is recommended, and
  14 days if you want weekly patterns.

The counts are saved in `data/traffic.db`.

---

## 7. Checking accuracy (manual count + evaluate)

This shows how close the automatic count is to a human count. You need it for the report.
**The target is less than 10 % error for cars in daylight.**

**Step 1.** In terminal 1, start the counter for one camera:

```bash
python -m traffic run --only 2
```

**Step 2.** In terminal 2, start counting by hand on the same camera:

```bash
python -m traffic manual-count 2
```

Watch the video. Every time something crosses the line, press a key:

| Key | Object |
|---|---|
| `c` | car |
| `t` | truck |
| `b` | bus |
| `m` | motorcycle |
| `p` | pedestrian |
| `y` | bicycle |

- **Lowercase** letter = direction **"in"** (the way the arrow points)
- **Uppercase** letter (with Shift) = direction **"out"**
- **Backspace** undoes the last key press, and **q** finishes.

Count for about **15 minutes**. When you quit, the result is saved to the `manual/` folder,
and the exact evaluate command to run next is printed.

**Step 3.** Compare:

```bash
python -m traffic evaluate manual/cam2_2026-10-20T08-00-00.csv
```

You get a table with your count, the automatic count and the error for each vehicle type,
plus whether the cars meet the 10 % target.

**Recommended:** do one count per camera in **daylight** and one at **night**. Then compare
them all at once and save a summary table:

```bash
python -m traffic evaluate manual/*.csv --summary reports/accuracy.csv
```

Optional settings for evaluate:

| Option | Meaning |
|---|---|
| `--start 2026-10-20T08:00:00Z --end 2026-10-20T08:15:00Z` | Compare only this time window (UTC) |
| `--lag 5` | Seconds of tolerance around your first and last key press (default 5) |
| `--summary file.csv` | Also save the summary table to a file |

---

## 8. Report – charts and tables for the written part

```bash
python -m traffic report                                        # last 7 days, all cameras
python -m traffic report --from 2026-10-20 --to 2026-10-26      # chosen dates
python -m traffic report --from 2026-10-20 --to 2026-10-26 --camera 2
python -m traffic report --camera 1 --camera 2 --day-hours 6-20
```

Everything is saved in `reports/<from>_<to>/`:

| File | Content |
|---|---|
| `hourly.png` | Traffic per hour over the whole period |
| `daily_profile.png` | Average traffic for each hour of the day |
| `heatmap.png` | Hour × day-of-week heatmap |
| `classes.png` | Vehicle types and directions |
| `periods.png` | Day vs night, weekday vs weekend |
| `decomposition.png` | Trend, daily and weekly pattern, anomalies (only with 2+ days of data) |
| `summary.txt` | The main numbers in plain text |
| `*.csv` | The same numbers as tables (open them in Excel) |

Options:

| Option | Meaning |
|---|---|
| `--from` / `--to` | First and last day (YYYY-MM-DD). Default: the last 7 days. |
| `--camera` | Only this camera. Repeat it for several cameras. |
| `--day-hours 7-19` | Which hours count as "day" |
| `--out folder` | Save somewhere other than `reports/` |

---

## 9. Recommended plan for the course project

1. **Setup** (section 2).
2. **Check the cameras.** Run `python -m traffic run --only 2 --preview` and make sure
   the line is in a good place. If not, re-run **Calibrate**.
3. **Start counting** with `python -m traffic run` and leave it running for **7–14 days**.
   The computer must stay on and must not go to sleep.
4. **During that week**, do the **manual counts** (section 7): each camera, day and night.
5. **Check progress** at any time with the **Report**.
6. **At the end**, run **Evaluate** with `--summary` and then **Report**, and put the
   pictures and tables into the written part.

---

## 10. Other commands

| Command | What it does |
|---|---|
| `python -m traffic --help` | Lists all commands |
| `python -m traffic run --help` | Help for one command (works for each tool) |
| `python -m traffic -c other.yaml run` | Use a different settings file |
| `pip install -r requirements-dev.txt` and then `pytest` | Run the automatic tests (developers) |

---

## 11. Common problems

| Problem | What to do |
|---|---|
| `pip install` fails with `No matching distribution found for torch==...` | Your Python is too new. Delete the environment (`rm -rf .venv`) and create it again with `python3.13 -m venv .venv` from the `v3` folder (section 2). |
| A camera shows "reconnecting" all the time | The stream is probably offline. Try it in a browser, or set `enabled: false` and use another camera. |
| The computer is too slow (`load` above 1, stride stuck at 10) | In `config.yaml`, use `backend: openvino` (x86), or run fewer cameras |
| Counts look wrong | Run with `--preview` and check the line. Re-run **Calibrate** if the camera view has changed. |
| The report says there is no database | Run the counter (`python -m traffic run`) at least once first |
| Empty places in charts | The camera was not working at that time. This is shown on purpose and is not zero traffic. |

---

## 12. Running 24/7 on a server (systemd)

On a Linux server the counter runs as a **systemd service** called `traffic`. It starts by
itself after a reboot and is restarted if it crashes. Do the one-time setup (section 2) in
`~/traffic-counter` first.

### The service file

Create `/etc/systemd/system/traffic.service` (replace `<user>` with your login name):

```ini
[Unit]
Description=Traffic counter
After=network-online.target
Wants=network-online.target

[Service]
KillMode=mixed
TimeoutStopSec=30
User=<user>
WorkingDirectory=/home/<user>/traffic-counter
Environment=OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
ExecStart=/home/<user>/traffic-counter/.venv/bin/python -m traffic run
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

What the important lines do:

| Line | Why it is needed |
|---|---|
| `KillMode=mixed` + `TimeoutStopSec=30` | On stop, only the main process gets the signal. It stops the cameras itself, so the last counts are saved and a `stopped` row is written. Without it every restart loses up to 2 s of counts, and the report counts up to 90 s after the stop as zero traffic. |
| `Environment=...` | Keeps the helper libraries to one thread each, and stops idle threads from wasting CPU. The program shares the CPU cores between the cameras by itself. |
| `ExecStart=... run` (no `--only`) | Runs every camera with `enabled: true` in `config.yaml`. |

Turn it on:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now traffic
systemctl is-enabled traffic           # must say "enabled"
```

If you change an existing service file with `sed`, check the result. A `sed` that replaces
a line does nothing if the line is missing:

```bash
sudo sed -i 's/^Environment=.*/Environment=OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1/' /etc/systemd/system/traffic.service
sudo sed -i 's/ --only [0-9,]*//' /etc/systemd/system/traffic.service
grep -A8 '^\[Service\]' /etc/systemd/system/traffic.service
sudo systemctl daemon-reload && sudo systemctl restart traffic
```

After **every** change to the service file, run `sudo systemctl daemon-reload` first.
Otherwise systemd keeps using the old version.

### Updating to a new version

```bash
cd ~/traffic-counter && git pull && sudo systemctl restart traffic
```

Keep `data/traffic.db`. Only delete it (with the service stopped) if you **renumbered the
camera ids** in `config.yaml`, because old rows would be shown under the wrong camera.

### Checking that it works

```bash
journalctl -u traffic -f                                          # live log (Ctrl+C stops watching, not the service)
journalctl -u traffic --since "1 hour ago" | grep -cE "reconnect|is down"   # should be 0 or close to it
sqlite3 data/traffic.db "select camera_id, count(*) from crossing_event group by 1"
```

Every 30 s each camera logs a line like:

```
camera 4: 11.9 analysed frames/s, stride 2, load 0.71, 3 active tracks
```

| Part | Meaning |
|---|---|
| `analysed frames/s` | frames the model looked at per second |
| `stride` | every how many frames one is analysed (1 = every frame) |
| `load` | busy time per second of video. **Below 1 = keeps up** with the live stream (the rest is waiting for the next video piece); above 1 = falls behind |

With `vid_stride: auto` (the default) the camera picks the stride by itself: the smallest one
whose predicted load stays under 0.8. It goes up at once when the camera gets too slow, and
down by at most one step per 30 s when there is spare time. So after a start it may take a
few minutes to settle. Nothing to tune.

The cameras deliver video in pieces of about 7.5 s: a whole piece arrives at once, then
nothing until the next one. That is why the camera does **not** just take the newest frame
(it would skip most of each piece), and why `load` is measured against video time, not
against the clock.

To use a fixed stride instead, set for example `vid_stride: 3` (every 3rd frame). Then the
camera must reach `stream fps / 3` analysed frames/s, e.g. 25 / 3 = 8.3. If it stays below
that and keeps falling, it is too slow: raise `vid_stride` or run fewer cameras. If it is
below but steady, the stream sends fewer frames than it says (camera 4 sends about 20 fps,
not 24.3).

Find a stream's fps without installing anything:

```bash
.venv/bin/python -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print(c.get(cv2.CAP_PROP_FPS))" <stream-url>
```

The total speed of the server is shared between the cameras. A higher stride can miss some
fast cars on a fast road, so do a short manual count (section 7) on the fastest camera.

### Faster model backend

`backend` in `config.yaml` chooses how the model runs. Measured on the AMD EPYC server
with 4 cameras at once (yolo26n, `imgsz: [384, 640]`):

| `backend` | analysed frames/s per camera |
|---|---|
| `pytorch` | 8.6 |
| `onnx` | 10-12 (needs `pip install onnx onnxruntime onnxslim`) |
| `openvino` (default) | 16 |

These numbers are the model alone; reading the stream costs a little more on top.
The first start after a change exports the model once (a few seconds), as for example
`yolo26n_384x640_openvino_model/`. Changing `imgsz` creates a new export by itself. If the
export fails, the log says so and the counter runs on `pytorch` instead.
OpenVINO is made for x86 processors (Intel / AMD). On an Apple Silicon Mac it is slower
than `pytorch`, so use `backend: pytorch` there if you run the counter on a Mac.

### One CPU core per camera (`pin_cores`)

On Linux each camera process is locked to its own physical CPU core (both of its
hyperthreads), for example camera 1 on CPUs 0 and 4. The log shows it:
`camera 1 started (..., CPUs 0,4, ...)`. Without this, OpenVINO in every camera picks the
same CPUs and the cameras crowd onto half of the CPUs (in `htop`: CPUs 4-7 at 100 %,
0-3 almost idle). Measured on the server with 4 cameras: 63.5 analysed frames/s in total
without, 77.4 with (+22 %). With fewer cameras than cores each camera gets several cores;
with more, they share. Set `pin_cores: false` to turn it off. On macOS it does nothing.

Lines like `[http @ 0x...] Error reading HTTP response: End of file` in the log come from the
video library when the camera server cuts one download short. They are retried by
themselves and are harmless unless they are followed by `stream ... is down`.

---

**Privacy:** the program never saves video or images and does not recognise faces or
number plates. Only counts are stored.
