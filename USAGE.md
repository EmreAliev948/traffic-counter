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
- You can view the results in a web dashboard or export charts for your report.

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
| **Dashboard** | Web page with numbers, charts and outages | `python -m traffic dashboard` |
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
- `url` – the stream address (a YouTube link or a direct `.m3u8` link)
- `enabled` – `true` = counted when you run the counter, `false` = skipped
- `line` – the counting line (filled in by the Calibrate tool)

### Adding a new camera

1. Add a new entry at the end of `config.yaml`:

   ```yaml
     - id: "20"
       name: Sofia - example street
       url: https://www.youtube.com/watch?v=XXXXXXXXXXX
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

## 7. Dashboard – view the results

```bash
python -m traffic dashboard
```

It opens in your browser at **http://localhost:8501**. You can run it while the counter is running.

In the left sidebar you choose:

- the **period** (dates)
- the **cameras**
- which hours count as **day** (default 07:00–19:00)

The dashboard shows:

| Section | What you see |
|---|---|
| Key numbers | Total crossings, "in" and "out" totals, how many hours the cameras actually worked, and the average per hour |
| Crossings per hour | A line per vehicle type. **A gap in the line means the camera was not working** (not zero traffic). |
| By class / by direction | Bar charts of vehicle types and directions |
| Heatmap | Busy hours for each day of the week. Empty cells = no data. |
| Comparisons | Day vs night and weekday vs weekend, plus the share of each vehicle type |
| Time-series breakdown | Splits traffic into the long-term trend, the daily pattern, the weekly pattern and what is left over. Needs at least 2 days of data (14 days for the weekly pattern). |
| Anomalies | Hours with unusually high or low traffic |
| Outages | When each camera was not working, and why (stream dropped, program stopped, …) |

---

## 8. Checking accuracy (manual count + evaluate)

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

## 9. Report – charts and tables for the written part

```bash
python -m traffic report                                        # last 7 days, all cameras
python -m traffic report --from 2026-10-20 --to 2026-10-26      # chosen dates
python -m traffic report --from 2026-10-20 --to 2026-10-26 --camera 2
python -m traffic report --camera 2 --camera 5 --day-hours 6-20
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

## 10. Recommended plan for the course project

1. **Setup** (section 2).
2. **Check the cameras.** Run `python -m traffic run --only 2 --preview` and make sure
   the line is in a good place. If not, re-run **Calibrate**.
3. **Start counting** with `python -m traffic run` and leave it running for **7–14 days**.
   The computer must stay on and must not go to sleep.
4. **During that week**, do the **manual counts** (section 8): each camera, day and night.
5. **Check progress** at any time with the **Dashboard**.
6. **At the end**, run **Evaluate** with `--summary` and then **Report**, and put the
   pictures and tables into the written part.

---

## 11. Other commands

| Command | What it does |
|---|---|
| `python -m traffic --help` | Lists all commands |
| `python -m traffic run --help` | Help for one command (works for each tool) |
| `python -m traffic -c other.yaml run` | Use a different settings file |
| `pip install -r requirements-dev.txt` and then `pytest` | Run the automatic tests (developers) |

---

## 12. Common problems

| Problem | What to do |
|---|---|
| `pip install` fails with `No matching distribution found for torch==...` | Your Python is too new. Delete the environment (`rm -rf .venv`) and create it again with `python3.13 -m venv .venv` from the `v3` folder (section 2). |
| A camera shows "reconnecting" all the time | The stream is probably offline. Try it in a browser, or set `enabled: false` and use another camera. |
| YouTube cameras stop opening | Update the downloader with `pip install -U yt-dlp` |
| The computer is too slow | In `config.yaml`, raise `vid_stride` (for example to 3) or run fewer cameras |
| Counts look wrong | Run with `--preview` and check the line. Re-run **Calibrate** if the camera view has changed. |
| The dashboard says there is no database | Run the counter (`python -m traffic run`) at least once first |
| Empty places in charts | The camera was not working at that time. This is shown on purpose and is not zero traffic. |

---

**Privacy:** the program never saves video or images and does not recognise faces or
number plates. Only counts are stored.
