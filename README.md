# ASL Fingerspelling Translator

Real-time American Sign Language alphabet -> text (and speech) from a webcam.

**Pipeline:** webcam frame -> MediaPipe Hand Landmarker (21 3-D keypoints) -> normalize
(wrist origin, unit scale, left hands mirrored) -> 73 features -> numpy MLP
(73-128-64-24, built from my CSE 4310 fc / ReLU / softmax / L2 layers) -> temporal smoothing
+ hold-to-type -> text, spoken with Windows speech.

| File | Role |
|---|---|
| `collect.py` | webcam recorder: hold a letter, press its key to save landmark frames |
| `train.py` | trains the MLP with validation on held-out recording takes, saves weights and plots |
| `translate.py` | live translator with smoothing, hold-to-type and speech |
| `common.py` | hand tracking, feature extraction, MLP |
| `nn_layers.py` | fully connected, ReLU, softmax loss and L2 layers written in NumPy |

## Run
```
pip install -r requirements.txt
python collect.py     # record your own signs: hold a letter, press its key (3+ takes per letter)
python train.py       # trains, writes models/asl_mlp.npz + results/*.png
python translate.py   # live translator
```
The MediaPipe hand landmarker model (`models/hand_landmarker.task`) downloads automatically on first run.
Your recordings (`data/`), trained weights (`models/`) and plots (`results/`) stay local.
Static letters only (A-Y without J and Z, which need motion).
Validation holds out whole recording takes, so the accuracy isn't inflated by near-duplicate frames.
