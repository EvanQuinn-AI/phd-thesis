# combat_tracker_bundle

Drop-in bundle of the tracker + recognizer + qualitative-eval modules,
ready to copy into another project that already has YOLOv26
(detection or segmentation) trained on combat-sports classes.

## What's inside

```
combat_tracker_bundle/
├── README.md                       (this file)
├── INTEGRATION.md                  Claude follows this when asked to integrate
├── requirements.txt                runtime + test deps
├── tracking/                       core tracker package
│   ├── pvp.py                      two-slot PvP tracker (Kalman + part-ReID)
│   ├── pve.py                      single-person + bag tracker (impact attr.)
│   ├── features.py                 pose-indexed mask-aware part histograms
│   ├── masks.py                    mask helpers + GrabCut + YOLO-seg adapter
│   ├── occlusion.py                clinch detection (mask-IoU when available)
│   ├── ownership.py                action ownership (mask_iou → kinematic → centroid)
│   ├── analytics.py                per-fighter throws / land / hit% / clinch / travel
│   ├── anchoring.py                first-N-frames identity anchor (single + dual)
│   ├── person_filter.py            referee / background suppression
│   ├── kalman.py                   constant-velocity bbox Kalman
│   ├── base.py                     Track + FeatureBank
│   ├── integration.py              opt-in adapter for existing PvP Streamlit apps
│   └── config.py                   thresholds dataclass
├── combat_tracker_recognizer/      subclass action recognition + active learning
│   ├── ...                         see combat_tracker_recognizer/README-style summary
├── tracking_eval/                  qualitative eval scripts
│   ├── run_qualitative.py          --mask-mode {none|yolo_seg|grabcut}
│   └── run_recognizer.py           empty bank → label clusters → replay → KNOWN
└── tests/tracking/                 79 pytests covering every module
```

## How to install

```bash
# Copy the bundle into your project root.
cp -r combat_tracker_bundle/* your_project/

# Install bundle deps on top of whatever you already have.
pip install -r your_project/requirements.txt
```

The bundle assumes Python 3.10+ and a working OpenCV. PyTorch is required
only if you use the optional `--mask-mode yolo_seg` path (which loads
your YOLOv26-seg model via Ultralytics).

## How to ask Claude to wire it up

Open this folder's `INTEGRATION.md` and copy-paste it (or its summary)
into your conversation. The file lists exactly which integration points
need adapting to your project's YOLOv26 model and class set.

The short version is:
1. Replace the legacy YOLOv5 loader in `tracking_eval/run_qualitative.py`
   with your project's YOLOv26 loader (one function — `_load_yolo`).
2. Update the class-id maps (`_PVP_CLASS_NAMES`, `_PERSON_CLASS_PVP`,
   `_BAG_CLASS_PVP`) to match your dataset's class indices.
3. Run the bundled tests; they should all pass without touching
   anything else.
4. Run `python tracking_eval/run_qualitative.py --mode pvp
   --mask-mode yolo_seg` against a sample clip to verify end-to-end.

## Test status before bundling

```
pytest tests/tracking/                  # 79 passed
pytest combat_tracker_recognizer/tests/  # 48 passed, 1 xpass
```
