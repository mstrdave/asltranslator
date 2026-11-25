"""
collect.py - record your own training data from the webcam.

  Press a letter key (a-z) while holding that ASL letter -> 1 s countdown, then
  ~2.5 s of hand keypoints are recorded (move your hand around a little: closer,
  farther, slight tilts - that variety makes the model robust).
  -    undo the last take
  ESC  quit (q is a letter!)

Aim for 3+ takes per letter. J and Z involve motion, so they're skipped
by the translator (static letters only).
Data is appended to data/landmarks.csv as: label, take_id, 63 normalized coordinates.
"""

import csv
import os
import time
from collections import Counter

import cv2
import numpy as np

from common import for_display, DATA_CSV, HandTracker, draw_hand, normalize, open_camera, put_text

FRAMES_PER_TAKE = 75
COUNTDOWN = 1.0


def load_counts():
    c, takes = Counter(), set()
    if os.path.exists(DATA_CSV):
        with open(DATA_CSV, newline="") as f:
            for row in csv.reader(f):
                c[row[0]] += 1
                takes.add(int(row[1]))
    return c, (max(takes) + 1 if takes else 0)


def main():
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    counts, take_id = load_counts()
    cap = open_camera()
    tracker = HandTracker(video=True)
    state, label, t_start, buf = "idle", None, 0.0, []
    last_take = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        frame = cv2.flip(frame, 1)  # mirror view, like a selfie
        pts, hand = tracker.detect(frame)
        frame, pts_d = for_display(frame, pts)
        if pts is not None:
            draw_hand(frame, pts_d)

        if state == "countdown":
            left = COUNTDOWN - (time.monotonic() - t_start)
            put_text(frame, f"'{label.upper()}' in {left:.1f}", (20, 80), 1.4, (0, 220, 255), 3)
            if left <= 0:
                state = "recording"
        elif state == "recording":
            if pts is not None:
                buf.append(normalize(pts, hand))
            n = len(buf)
            put_text(frame, f"REC '{label.upper()}' {n}/{FRAMES_PER_TAKE}", (20, 80), 1.4, (60, 60, 255), 3)
            cv2.rectangle(frame, (20, 95), (20 + int(400 * n / FRAMES_PER_TAKE), 105), (60, 60, 255), -1)
            if n >= FRAMES_PER_TAKE:
                with open(DATA_CSV, "a", newline="") as f:
                    w = csv.writer(f)
                    for p in buf:
                        w.writerow([label, take_id, *np.round(p.reshape(-1), 5)])
                counts[label] += n
                last_take = (take_id, label, n)
                take_id += 1
                state, buf = "idle", []
        else:
            put_text(frame, "Hold a sign, press its letter key  |  '-' undo  |  ESC quit", (20, 40), 0.65)
            if pts is None:
                put_text(frame, "no hand detected", (20, 75), 0.65, (0, 160, 255))

        # per-letter sample counts along the bottom
        h = frame.shape[0]
        for k, ch in enumerate("abcdefghiklmnopqrstuvwxy"):
            n = counts.get(ch, 0)
            col = (90, 220, 90) if n >= 200 else (0, 200, 255) if n > 0 else (150, 150, 150)
            put_text(frame, f"{ch.upper()} {n}", (12 + (k % 12) * 78, h - 45 + (k // 12) * 28), 0.55, col, 1)

        cv2.imshow("ASL data collection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if state == "idle" and ord("a") <= k <= ord("z"):
            label, state, t_start = chr(k), "countdown", time.monotonic()
        elif state == "idle" and k == ord("-") and last_take:
            tid, lab, n = last_take
            with open(DATA_CSV, newline="") as f:
                rows = [r for r in csv.reader(f) if int(r[1]) != tid]
            with open(DATA_CSV, "w", newline="") as f:
                csv.writer(f).writerows(rows)
            counts[lab] -= n
            last_take = None
            print(f"removed take {tid} ({lab.upper()})")

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Samples per letter:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
