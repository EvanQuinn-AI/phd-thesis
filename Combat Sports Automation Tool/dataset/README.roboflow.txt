
CombatSports - v7 version-seven
==============================

This dataset was exported via roboflow.com on June 17, 2024 at 9:26 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 5261 images.
Combat-sports are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 15 percent of the image
* Random rotation of between -3 and +3 degrees
* Random shear of between -6° to +6° horizontally and -3° to +3° vertically
* Random brigthness adjustment of between -3 and +3 percent
* Random exposure adjustment of between -9 and +9 percent
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 0.15 percent of pixels


