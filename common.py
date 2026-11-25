"""
common.py - shared pieces of the ASL fingerspelling translator.

  HandTracker   MediaPipe Hand Landmarker -> 21 (x, y, z) keypoints per frame
  normalize()   makes keypoints position/scale/handedness invariant
  features()    normalized keypoints + fingertip distances -> 73-dim vector
  MLP           small numpy neural net built from the CSE 4310 layers (nn_layers.py)
  draw_hand()   skeleton overlay for OpenCV frames
"""

import os
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from nn_layers import fc_backward, fc_forward, l2_regularization, relu_backward, relu_forward, softmax_loss

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_TASK = os.path.join(HERE, "models", "hand_landmarker.task")
DATA_CSV = os.path.join(HERE, "data", "landmarks.csv")
WEIGHTS = os.path.join(HERE, "models", "asl_mlp.npz")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")

TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky fingertips
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
               (17, 18), (18, 19), (19, 20)]


# --------------------------------------------------------------------------
# Hand tracking
# --------------------------------------------------------------------------
class HandTracker:
    def __init__(self, video=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        if not os.path.exists(MODEL_TASK):
            import urllib.request
            os.makedirs(os.path.dirname(MODEL_TASK), exist_ok=True)
            print("downloading MediaPipe hand landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_TASK)
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=MODEL_TASK), running_mode=mode, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6, min_tracking_confidence=0.5)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.video = video
        self.t0 = time.monotonic()
        self.last_ts = -1

    def detect(self, bgr):
        """Returns (pts (21,3) in pixel-ish units, 'Left'/'Right') or (None, None)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = max(self.last_ts + 1, int((time.monotonic() - self.t0) * 1000))
            self.last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)
        if not res.hand_landmarks:
            return None, None
        h, w = bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.hand_landmarks[0]], dtype=np.float32)
        return pts, res.handedness[0][0].category_name

    def close(self):
        self.lm.close()


def normalize(pts, handedness):
    """Wrist at origin, unit size, left hands mirrored to look like right hands."""
    p = pts - pts[0]
    p = p / (np.linalg.norm(p[:, :2], axis=1).max() + 1e-6)
    if handedness == "Left":
        p[:, 0] *= -1
    return p.astype(np.float32)


def features(norm_pts):
    """(21,3) normalized points -> 73-dim feature: 63 coords + 10 fingertip-pair distances."""
    t = norm_pts[TIPS]
    d = [np.linalg.norm(t[i] - t[j]) for i in range(5) for j in range(i + 1, 5)]
    return np.concatenate([norm_pts.reshape(-1), np.array(d, np.float32)])


def augment(norm_pts, rng):
    """Random in-plane rotation, scale and jitter for training robustness."""
    a = rng.uniform(-0.3, 0.3)  # about +-17 degrees
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
    p = norm_pts @ R.T * rng.uniform(0.9, 1.1)
    p[:, 2] *= rng.uniform(0.7, 1.3)
    p = p + rng.normal(0, 0.012, p.shape).astype(np.float32)
    p[0] = 0  # the wrist is the origin by definition; jittering it explodes its ~0 std-dev
    return p


# --------------------------------------------------------------------------
# Classifier: 73 -> 128 -> 64 -> C MLP using the hand-written layers
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, din, classes, hidden=(128, 64), seed=0):
        rng = np.random.default_rng(seed)
        dims = [din, *hidden, len(classes)]
        self.classes = list(classes)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]
        self.mu = np.zeros(din)
        self.sd = np.ones(din)

    def scores(self, X):
        h = (X - self.mu) / self.sd
        caches = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h, c_fc = fc_forward(h, W, b)
            c_relu = None
            if i < len(self.W) - 1:
                h, c_relu = relu_forward(h)
            caches.append((c_fc, c_relu))
        return h, caches

    def loss(self, X, y, reg):
        s, caches = self.scores(X)
        loss, g = softmax_loss(s, y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            c_fc, c_relu = caches[i]
            if c_relu is not None:
                g = relu_backward(g, c_relu)
            g, gW[i], gb[i] = fc_backward(g, c_fc)
            rl, rg = l2_regularization(self.W[i], reg)
            loss += rl
            gW[i] = gW[i] + rg
        return loss, gW, gb

    def predict_proba(self, X):
        s, _ = self.scores(np.atleast_2d(X))
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path=WEIGHTS):
        np.savez(path, classes=np.array(self.classes), mu=self.mu, sd=self.sd,
                 **{f"W{i}": W for i, W in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path=WEIGHTS):
        d = np.load(path)
        n = sum(1 for k in d.files if k.startswith("W"))
        m = cls.__new__(cls)
        m.classes = [str(c) for c in d["classes"]]
        m.W = [d[f"W{i}"] for i in range(n)]
        m.b = [d[f"b{i}"] for i in range(n)]
        m.mu, m.sd = d["mu"], d["sd"]
        return m


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_hand(frame, pts, color=(80, 220, 200)):
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(int(v) for v in pts[a, :2]), tuple(int(v) for v in pts[b, :2]), color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in TIPS else 3
        cv2.circle(frame, (int(p[0]), int(p[1])), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(p[0]), int(p[1])), r, color, 1, cv2.LINE_AA)


def put_text(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    # 640x480 keeps most webcams at full frame rate; we upscale only for display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


DISPLAY_H = 720


def for_display(frame, pts):
    """Upscale the camera frame (and keypoints) so overlays are readable on screen."""
    f = DISPLAY_H / frame.shape[0]
    big = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return big, (pts * np.array([f, f, f], np.float32) if pts is not None else None)
