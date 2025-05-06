# Phd-Thesis Project Breakdown
The dataset used throughout this thesis is publicly archived as Combat Sports Dataset on Zenodo (https://doi.org/10.5281/zenodo.15349809). A permanent landing page with download links, version history and metadata is available at the DOI above. The original workspace, tooling notes and optional export formats remain mirrored on Roboflow Universe at https://universe.roboflow.com/combatsports/combatsports-merge-attempt/dataset/2

Dataset Overview
Item	Value
Total images	7 757
Classes	boxing bag, cross, high guard, hook, kick, low guard, person
Annotation format	YOLO v5 (one .txt per image)
Resolution	640 × 640 px (stretched)
Licence	Public Domain (CC0)
Curator	Evan Quinn, 2025 01 08 22:52 UTC

Directory Map
├── train/
│   └── images/   (6906 *.jpg)
├── valid/
│   └── images/   (663 *.jpg)
└── test/
    └── images/   (188 *.jpg)

Pre processing pipeline
These steps ensure uniform dimensions and dynamic range for YOLO input. Before augmentation every frame underwent:
1.	Auto orientation (EXIF corrected and stripped)
2.	Resize to a fixed square canvas of 640 × 640 px using stretch mode
3.	Contrast stretching with 2–98 % percentile clipping
Each source image is replicated three times with randomised transformations drawn from the ranges below, expanding the effective training set to 31 028 images.
Augmentation	Range / probability
Random crop (zoom)	0 – 15 %
Rotation	–3° … +3°
Shear	horizontal –6° … +6°, vertical –3° … +3°
Greyscale conversion	applied to 25 % of images
Hue shift	–4° … +4°
Saturation shift	–5 % … +5 %
Brightness shift	–3 % … +3 %
Exposure shift	–9 % … +9 %
Gaussian blur	0 … 2 px radius
Salt and pepper noise	0.15 % of pixels

Class Definitions
These classes were featured throughout the thesis and are generalised to be easily understood by anyone in reading the thesis.
ID	Name	Description
 0	boxing bag	Heavy bag target present in training footage
 1	cross	Straight hand punch or jab
 2	high guard	Hands protecting head, elbows tucked
 3	hook	Circular punch delivered to side of target
 4	kick	Any lower  or upper body striking kick
 5	low guard	Guard dropped below chin level
 6	person	Athlete’s full silhouette

Ethical and Legal Statement
All footage was recorded in controlled gym environments with written or verbal consent from participating athletes. Faces of by standers were cropped out or blurred where necessary; no minors appear in the released frames. Because the dataset is dedicated to the public domain, users may copy, modify and redistribute it without restriction, but citation of the Zenodo DOI is requested to acknowledge provenance.
Reproduction Note
The training, validation and test images above are exactly those referenced in Chapters 4–6. Clone the thesis code repository, set the DATA_DIR environment variable to the directory structure shown in the Directory Map, and run train_yolo.py --config configs/yolo.yaml to reproduce baseline results.
GitHub Code Repository
1.	Clone repo ➜ git clone https://github.com/EvanQuinn-AI/thesis-projects.git
2.	Create virtual env ➜ conda env create -f setup-files/environment.yml or pip install -r requirements
3.	Download dataset to dataset/ and weights to models/ (setup script offers prompts).
4.	Train ➜ python gpu-version/train.py --config configs/yolo.yaml
5.	Infer (CPU) ➜ double click run_cpu_app.bat or run python cpu-version/infer.py --source myvideo.mp4
6.	Results and TensorBoard logs appear under runs/experiment <timestamp>/.
This separation of CPU and GPU sub projects keeps device specific dependencies isolated, while the common models/, dataset/, and runs/ directories make it easy to share checkpoints and compare outputs across hardware configurations.

Ensure Project Structure Follows this example, if the app doesnt run, try pip install -r requirements.txt from within the project directory containing the file; or alternatively edit the batch file using textpad to change directory/filenames to match.

![image](https://github.com/user-attachments/assets/5d67eac1-010d-4745-96ff-0193bb79917e)
