# Terraria Auto-Fishing Helper

Screen-motion-based auto-fishing helper for Terraria. The script watches a small region around your bobber and reels/recasts when it detects a bite-like motion.

## Features

- Hotkeys to calibrate, start/pause, and stop
- Motion detection using frame-to-frame and baseline differences
- Drift tolerance (`--search-margin`) to catch bobber bounces away from the calibrated spot
- Windows safety check to avoid clicking when Terraria is not the active window

## Requirements

- Python 3.10+ (recommended)
- Terraria running in windowed or borderless mode (more reliable than fullscreen)

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Files

- `terraria_autofish.py`: main script
- `requirements.txt`: Python dependencies

## Quick Start

Run the script:

```powershell
python terraria_autofish.py
```

In-game usage:

1. Stand where you want to fish and get ready to cast.
2. Move your mouse cursor over the bobber (or where the bobber usually lands).
3. Press `i` to calibrate the detection region.
4. Press `o` to start auto-fishing.
5. Press `o` again to pause.
6. Press `p` to exit the script.

## Hotkeys

- `i`: Calibrate bobber region (uses current mouse position)
- `o`: Start / pause automation
- `p`: Exit script

## How It Works (Short Version)

- The script captures a small square region around the calibrated point.
- It compares the current frame to:
  - the previous frame (sudden movement)
  - a rolling baseline (change from normal bobber state)
- If motion crosses both thresholds for enough consecutive frames, it reels and recasts.
- If no bite is detected for a while, it forces a recast.

## Important Notes

- Keep the cast area stable (camera movement and large background motion can cause false triggers).
- On Windows, clicks are only sent when the active window title starts with `Terraria`.
- If you alt-tab away, detection pauses until Terraria is focused again.

## Tuning / Common Issues

### Bobber bounces away from the usual spot

Increase `--search-margin` so the script scans a larger area around your calibrated point.

Examples:

```powershell
python terraria_autofish.py --search-margin 16
python terraria_autofish.py --search-margin 20
```

Tradeoff: larger values can increase false triggers if the background has lots of motion.

### It misses bites

Try one or more of:

- Increase `--search-margin`
- Lower `--prev-diff-threshold`
- Lower `--base-diff-threshold`
- Increase `--roi-size` slightly (can help if the bobber moves more)
- Run with `--verbose` to inspect live diff values

Example:

```powershell
python terraria_autofish.py --verbose
```

### It triggers too often / false positives

Try one or more of:

- Decrease `--search-margin`
- Increase `--prev-diff-threshold`
- Increase `--base-diff-threshold`
- Increase `--trigger-frames` (requires more consecutive frames before triggering)

## CLI Options

You can view all options with:

```powershell
python terraria_autofish.py --help
```

Main options:

- `--roi-size`: Square capture size (pixels)
- `--search-margin`: Extra pixels around the calibrated spot to scan for bobber drift/bounce
- `--fps`: Capture FPS
- `--settle-seconds`: Ignore motion right after casting
- `--click-hold-ms`: Mouse button hold duration per click
- `--post-reel-delay`: Delay after reel click before recast
- `--post-cast-delay`: Delay after cast before detection resumes
- `--idle-recast-seconds`: Force recast if no bite is detected
- `--prev-diff-threshold`: Frame-to-frame motion threshold
- `--base-diff-threshold`: Baseline difference threshold
- `--trigger-frames`: Consecutive frames needed to trigger
- `--cooldown-seconds`: Minimum time between triggers
- `--button {left,right}`: Mouse button used for fishing
- `--verbose`: Print debug diff values

## Example Configs

Default:

```powershell
python terraria_autofish.py
```

More bobber drift tolerance:

```powershell
python terraria_autofish.py --search-margin 18
```

Conservative (fewer false triggers):

```powershell
python terraria_autofish.py --prev-diff-threshold 18 --base-diff-threshold 15 --trigger-frames 3
```
