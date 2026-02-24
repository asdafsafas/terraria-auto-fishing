#!/usr/bin/env python3
"""
Terraria auto-fishing helper (screen-motion based).

Hotkeys
- i: calibrate bobber region (move mouse over the bobber / expected bobber spot)
- o: start/pause automation
- p: exit script

Notes
- Works best in windowed or borderless mode with a stable camera/cursor position.
- This script watches a small screen region and triggers when it sees a sudden change.
- You will likely need to tune thresholds for your lighting/water/background.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mss import mss
from pynput import keyboard as pynput_keyboard
from pynput import mouse as pynput_mouse


@dataclass
class Config:
    roi_size: int = 28
    search_margin: int = 12
    fps: float = 30.0
    settle_seconds: float = 0.9
    click_hold_seconds: float = 0.025
    post_reel_delay: float = 1.20
    post_cast_delay: float = 1.00
    idle_recast_seconds: float = 45.0
    prev_diff_threshold: float = 14.0
    base_diff_threshold: float = 12.0
    trigger_frames: int = 2
    cooldown_seconds: float = 0.75
    button: str = "left"
    verbose: bool = False


class TerrariaAutoFisher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._enabled = False
        self._exit_requested = False
        self._roi_center: Optional[tuple[int, int]] = None
        self._last_hotkey: dict[str, float] = {}
        self._dpi_awareness_set = False
        self._dpi_warning_logged = False
        self._focus_api_warning_logged = False

        self._enable_windows_dpi_awareness()
        self._sct = mss()
        self._mouse = pynput_mouse.Controller()
        self._button = (
            pynput_mouse.Button.left
            if cfg.button == "left"
            else pynput_mouse.Button.right
        )

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    def _debounced(self, name: str, window: float = 0.25) -> bool:
        now = time.monotonic()
        last = self._last_hotkey.get(name, 0.0)
        if now - last < window:
            return False
        self._last_hotkey[name] = now
        return True

    def _get_roi_center(self) -> Optional[tuple[int, int]]:
        with self._lock:
            return self._roi_center

    def _set_roi_center(self, x: int, y: int) -> None:
        with self._lock:
            self._roi_center = (x, y)

    def _is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def _set_enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value

    def _request_exit(self) -> None:
        with self._lock:
            self._exit_requested = True
            self._enabled = False

    def _should_exit(self) -> bool:
        with self._lock:
            return self._exit_requested

    def _current_mouse_position(self) -> tuple[int, int]:
        x, y = self._mouse.position
        return int(x), int(y)

    def _click(self) -> None:
        if not self._is_terraria_foreground():
            self._log("[WARN] Click skipped because Terraria is not the active window")
            return
        # Some games miss extremely short synthetic clicks; hold briefly.
        self._mouse.press(self._button)
        time.sleep(self.cfg.click_hold_seconds)
        self._mouse.release(self._button)

    def _enable_windows_dpi_awareness(self) -> None:
        if os.name != "nt":
            return
        try:
            user32 = ctypes.windll.user32
            try:
                # Per-monitor DPI awareness when available.
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                # Fallback for older Windows APIs.
                user32.SetProcessDPIAware()
            self._dpi_awareness_set = True
        except Exception as exc:
            if not self._dpi_warning_logged:
                self._log(f"[WARN] Could not enable DPI awareness: {exc}")
                self._dpi_warning_logged = True

    def _active_window_title(self) -> str:
        if os.name != "nt":
            return ""
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ""
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, len(buf))
            return buf.value
        except Exception as exc:
            if not self._focus_api_warning_logged:
                self._log(f"[WARN] Foreground-window check unavailable: {exc}")
                self._focus_api_warning_logged = True
            return ""

    def _is_terraria_foreground(self) -> bool:
        if os.name != "nt":
            return True
        title = self._active_window_title()
        if not title:
            # Fail closed on Windows to avoid accidental clicks into other apps.
            return False
        # Use a strict prefix so terminal/editor titles containing the project
        # folder name ("terraria") do not count as the game window.
        return title.lower().startswith("terraria")

    def _grab_gray_region(self, size: int) -> np.ndarray:
        center = self._get_roi_center()
        if center is None:
            raise RuntimeError("ROI is not calibrated")

        x, y = center
        half = size // 2
        monitor = {
            "left": int(x - half),
            "top": int(y - half),
            "width": int(size),
            "height": int(size),
        }

        shot = self._sct.grab(monitor)
        arr = np.asarray(shot, dtype=np.uint8)
        # mss returns BGRA. Convert to grayscale float32 for stable diff math.
        gray = (
            0.114 * arr[:, :, 0].astype(np.float32)
            + 0.587 * arr[:, :, 1].astype(np.float32)
            + 0.299 * arr[:, :, 2].astype(np.float32)
        )
        return gray

    def _grab_gray_roi(self) -> np.ndarray:
        size = self.cfg.roi_size + (2 * self.cfg.search_margin)
        return self._grab_gray_region(size)

    @staticmethod
    def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def _windowed_motion_diffs(
        self, frame: np.ndarray, prev_frame: np.ndarray, baseline_frame: np.ndarray
    ) -> tuple[float, float]:
        """Return the strongest local motion score in a search region.

        When `search_margin` > 0, this scans all `roi_size x roi_size` windows and
        uses the window with the highest combined frame-to-frame / baseline change.
        This helps when the bobber temporarily bounces away from the calibrated spot.
        """
        roi = self.cfg.roi_size
        h, w = frame.shape
        if h == roi and w == roi:
            return (
                self._mean_abs_diff(frame, prev_frame),
                self._mean_abs_diff(frame, baseline_frame),
            )

        prev_abs = np.abs(frame - prev_frame)
        base_abs = np.abs(frame - baseline_frame)

        prev_windows = np.lib.stride_tricks.sliding_window_view(prev_abs, (roi, roi))
        base_windows = np.lib.stride_tricks.sliding_window_view(base_abs, (roi, roi))
        prev_means = prev_windows.mean(axis=(-2, -1))
        base_means = base_windows.mean(axis=(-2, -1))

        # Favor windows that are strong on both metrics, not just one.
        score = prev_means + base_means
        best_idx = np.unravel_index(int(np.argmax(score)), score.shape)
        return float(prev_means[best_idx]), float(base_means[best_idx])

    def _reel_and_recast(self, reason: str) -> None:
        self._log(f"[ACTION] {reason}: reel + recast")
        self._click()  # reel / collect
        time.sleep(self.cfg.post_reel_delay)
        if not self._is_enabled() or self._should_exit():
            return
        self._click()  # cast again
        time.sleep(self.cfg.post_cast_delay)

    def _start_cast(self) -> None:
        self._log("[ACTION] initial cast")
        self._click()
        time.sleep(self.cfg.post_cast_delay)

    def _autofish_loop(self) -> None:
        if self._get_roi_center() is None:
            self._log("[WARN] No ROI calibrated. Press i with mouse on bobber area.")
            self._set_enabled(False)
            return

        frame_interval = 1.0 / max(self.cfg.fps, 1.0)
        last_trigger_time = 0.0

        self._start_cast()
        cast_time = time.monotonic()
        prev_frame: Optional[np.ndarray] = None
        baseline_frame: Optional[np.ndarray] = None
        streak = 0

        while self._is_enabled() and not self._should_exit():
            loop_started = time.monotonic()
            if not self._is_terraria_foreground():
                # Pause detection while tabbed out so we do not detect browser/editor
                # motion and so idle timeout does not immediately recast on return.
                cast_time = loop_started
                prev_frame = None
                baseline_frame = None
                streak = 0
                time.sleep(0.10)
                continue

            try:
                frame = self._grab_gray_roi()
            except Exception as exc:
                self._log(f"[ERROR] Screen capture failed: {exc}")
                self._set_enabled(False)
                break

            if prev_frame is None:
                prev_frame = frame
                baseline_frame = frame.copy()
                time.sleep(frame_interval)
                continue

            prev_diff, base_diff = self._windowed_motion_diffs(
                frame, prev_frame, baseline_frame
            )
            now = time.monotonic()

            if now - cast_time < self.cfg.settle_seconds:
                # Let cast splash / bobber landing settle before detecting bites.
                prev_frame = frame
                baseline_frame = 0.9 * baseline_frame + 0.1 * frame
                time.sleep(frame_interval)
                continue

            triggered = (
                prev_diff >= self.cfg.prev_diff_threshold
                and base_diff >= self.cfg.base_diff_threshold
            )
            streak = streak + 1 if triggered else 0

            if self.cfg.verbose:
                self._log(
                    f"[DEBUG] prev={prev_diff:.2f} base={base_diff:.2f} "
                    f"streak={streak}"
                )

            calm = (
                prev_diff < self.cfg.prev_diff_threshold * 0.60
                and base_diff < self.cfg.base_diff_threshold * 0.60
            )
            if calm:
                baseline_frame = 0.92 * baseline_frame + 0.08 * frame

            if (
                streak >= self.cfg.trigger_frames
                and now - last_trigger_time >= self.cfg.cooldown_seconds
            ):
                self._reel_and_recast(
                    f"bite detected (prev={prev_diff:.1f}, base={base_diff:.1f})"
                )
                last_trigger_time = time.monotonic()
                cast_time = last_trigger_time
                prev_frame = None
                baseline_frame = None
                streak = 0
                continue

            if now - cast_time >= self.cfg.idle_recast_seconds:
                self._reel_and_recast("timeout recast")
                last_trigger_time = time.monotonic()
                cast_time = last_trigger_time
                prev_frame = None
                baseline_frame = None
                streak = 0
                continue

            prev_frame = frame
            elapsed = time.monotonic() - loop_started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def _on_press(self, key: pynput_keyboard.Key | pynput_keyboard.KeyCode) -> None:
        try:
            key_char = getattr(key, "char", None)
            if isinstance(key_char, str):
                key_char = key_char.lower()

            if key_char == "i" and self._debounced("calibrate"):
                x, y = self._current_mouse_position()
                self._set_roi_center(x, y)
                self._log(f"[INFO] ROI calibrated at ({x}, {y})")
                return

            if key_char == "o" and self._debounced("toggle"):
                new_state = not self._is_enabled()
                self._set_enabled(new_state)
                self._log("[INFO] Auto-fishing ON" if new_state else "[INFO] Auto-fishing PAUSED")
                return

            if key_char == "p" and self._debounced("exit"):
                self._log("[INFO] Exit requested")
                self._request_exit()
                return
        except Exception as exc:
            self._log(f"[ERROR] Hotkey handler failed: {exc}")
            self._request_exit()

    def run(self) -> None:
        self._log("Terraria auto-fishing helper")
        self._log("Hotkeys: i=calibrate ROI, o=start/pause, p=exit")
        self._log(
            "Usage: cast spot should remain stable; place cursor over bobber area and press i."
        )
        if os.name == "nt":
            self._log(
                "[INFO] Windows safeguards: DPI awareness "
                f"{'enabled' if self._dpi_awareness_set else 'not enabled'}, "
                "clicks require Terraria window focus."
            )

        listener = pynput_keyboard.Listener(on_press=self._on_press)
        listener.start()
        try:
            while not self._should_exit():
                if not self._is_enabled():
                    time.sleep(0.05)
                    continue
                self._autofish_loop()
                time.sleep(0.05)
        except KeyboardInterrupt:
            self._log("[INFO] KeyboardInterrupt")
        finally:
            self._request_exit()
            listener.stop()
            listener.join(timeout=1.0)
            self._log("[INFO] Script exited")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Screen-motion-based Terraria auto-fishing helper."
    )
    parser.add_argument("--roi-size", type=int, default=28, help="Square capture size in pixels.")
    parser.add_argument(
        "--search-margin",
        type=int,
        default=12,
        help=(
            "Extra pixels around the calibrated spot to scan for bobber movement. "
            "Higher values help catch bobber drift/bounce but may add noise."
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Capture FPS.")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.9,
        help="Ignore motion right after a cast/recast.",
    )
    parser.add_argument(
        "--click-hold-ms",
        type=float,
        default=100.0,
        help="How long to hold the mouse button down per click (milliseconds).",
    )
    parser.add_argument(
        "--post-reel-delay",
        type=float,
        default=1.20,
        help="Delay between reel click and recast click.",
    )
    parser.add_argument(
        "--post-cast-delay",
        type=float,
        default=1.00,
        help="Delay after cast before normal loop timing resumes.",
    )
    parser.add_argument(
        "--idle-recast-seconds",
        type=float,
        default=45.0,
        help="Force recast if no bite is detected for this long.",
    )
    parser.add_argument(
        "--prev-diff-threshold",
        type=float,
        default=14.0,
        help="Frame-to-frame motion threshold.",
    )
    parser.add_argument(
        "--base-diff-threshold",
        type=float,
        default=12.0,
        help="Difference from rolling baseline threshold.",
    )
    parser.add_argument(
        "--trigger-frames",
        type=int,
        default=2,
        help="Consecutive frames above threshold before triggering.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.75,
        help="Minimum time between triggers.",
    )
    parser.add_argument(
        "--button",
        choices=["left", "right"],
        default="left",
        help="Mouse button used for fishing action.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame diff values for threshold tuning.",
    )
    ns = parser.parse_args()

    roi_size = max(8, int(ns.roi_size))
    if roi_size % 2 != 0:
        roi_size += 1

    return Config(
        roi_size=roi_size,
        search_margin=max(0, int(ns.search_margin)),
        fps=max(1.0, float(ns.fps)),
        settle_seconds=max(0.0, float(ns.settle_seconds)),
        click_hold_seconds=max(0.0, float(ns.click_hold_ms)) / 1000.0,
        post_reel_delay=max(0.0, float(ns.post_reel_delay)),
        post_cast_delay=max(0.0, float(ns.post_cast_delay)),
        idle_recast_seconds=max(1.0, float(ns.idle_recast_seconds)),
        prev_diff_threshold=max(0.1, float(ns.prev_diff_threshold)),
        base_diff_threshold=max(0.1, float(ns.base_diff_threshold)),
        trigger_frames=max(1, int(ns.trigger_frames)),
        cooldown_seconds=max(0.0, float(ns.cooldown_seconds)),
        button=ns.button,
        verbose=bool(ns.verbose),
    )


def main() -> None:
    cfg = parse_args()
    fisher = TerrariaAutoFisher(cfg)
    fisher.run()


if __name__ == "__main__":
    main()
