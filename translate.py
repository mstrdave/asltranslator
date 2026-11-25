"""
translate.py - live ASL fingerspelling -> text (and speech).

Hold a letter steady to type it. Drop your hand for ~1 s to add a space.
To type a double letter (the LL in HELLO), lower your hand briefly or hold the letter longer.

Keys:  SPACE space   BACKSPACE delete   c clear   s speak sentence   ESC quit
"""

import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

from common import for_display, MLP, HandTracker, draw_hand, features, normalize, open_camera, put_text

CONF = 0.70        # minimum probability to accept a letter
HOLD = 0.45        # seconds a letter must be held to type it
REPEAT = 1.4       # holding the same letter this long types it again
SPACE_AFTER = 1.0  # seconds with no hand -> space
SMOOTH = 5         # frames of probability averaging


def speak(text):
    if not text.strip():
        return
    safe = text.replace("'", "''")
    cmd = ("Add-Type -AssemblyName System.Speech; "
           f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')")
    threading.Thread(target=subprocess.run, args=(["powershell", "-NoProfile", "-Command", cmd],),
                     kwargs={"capture_output": True}, daemon=True).start()


def main():
    try:
        model = MLP.load()
    except FileNotFoundError:
        print("No trained model yet - run collect.py, then train.py.")
        return
    cap = open_camera()
    tracker = HandTracker(video=True)
    probs = deque(maxlen=SMOOTH)
    text = ""
    cand, cand_since, typed_at = None, 0.0, 0.0
    no_hand_since = time.monotonic()
    fps_t, fps = time.monotonic(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        frame = cv2.flip(frame, 1)
        now = time.monotonic()
        pts, hand = tracker.detect(frame)
        frame, pts_d = for_display(frame, pts)
        letter, conf = None, 0.0

        if pts is not None:
            draw_hand(frame, pts_d)
            probs.append(model.predict_proba(features(normalize(pts, hand)))[0])
            p = np.mean(probs, axis=0)
            i = int(np.argmax(p))
            letter, conf = model.classes[i].upper(), float(p[i])
            no_hand_since = None

            if conf >= CONF:
                if letter != cand:
                    cand, cand_since, typed_at = letter, now, 0.0
                elif typed_at == 0.0 and now - cand_since >= HOLD:
                    text += letter
                    typed_at = now
                elif typed_at and now - typed_at >= REPEAT:
                    text += letter
                    typed_at = now
            else:
                cand = None

            # label near the hand
            x0, y0 = int(pts_d[:, 0].min()), int(pts_d[:, 1].min()) - 15
            col = (90, 220, 90) if conf >= CONF else (0, 180, 255)
            put_text(frame, f"{letter} {conf * 100:.0f}%", (x0, max(30, y0)), 1.0, col, 2)
        else:
            probs.clear()
            cand = None
            if no_hand_since is None:
                no_hand_since = now
            if now - no_hand_since >= SPACE_AFTER and text and not text.endswith(" "):
                text += " "

        # hold-progress ring
        if cand and typed_at == 0.0:
            frac = min(1.0, (now - cand_since) / HOLD)
            cv2.ellipse(frame, (60, 150), (34, 34), -90, 0, 360 * frac, (90, 220, 90), 6, cv2.LINE_AA)
            put_text(frame, cand, (45, 163), 1.1, (255, 255, 255), 2)

        # bottom text panel
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 24), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        shown = text[-32:] + ("_" if int(now * 2) % 2 else " ")
        put_text(frame, shown, (20, h - 38), 1.2, (255, 255, 255), 2)
        put_text(frame, "SPACE space | BKSP delete | c clear | s speak | ESC quit", (20, h - 10), 0.5,
                 (170, 170, 170), 1)

        dt = now - fps_t
        fps_t = now
        fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)
        put_text(frame, f"{fps:4.1f} fps", (w - 120, 30), 0.6, (200, 200, 200), 1)

        cv2.imshow("ASL Fingerspelling Translator", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 32:
            text += " "
        elif k == 8:
            text = text[:-1]
        elif k == ord("c"):
            text = ""
        elif k == ord("s"):
            speak(text)

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if text.strip():
        print("Final text:", text)


if __name__ == "__main__":
    main()
