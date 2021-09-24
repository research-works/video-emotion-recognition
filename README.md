## Pre-trained CNN models as feature extractors for Video Emotion Recognition

**What this is about ?**

This repository contains the work related to the research paper titled *Efficiency analysis of Pre-trained CNN models as feature extractors for Video Emotion Recognition*.

Following are the names of Authors:
- *Diksha Mehta*
- *Janhvi Joshi*
- *Abhishek Bisht* 
- *Pankaj Badoni*

Appropriate citations can be downloaded from the github page of this repository or alternatively from `citations.cff`.

**How to use this work ?**

- Two configuration files `config.ini` and `fusion_training_hyparam.json` are required under the `utils` directory for this 
project to run. The details of these files can be inferred from source code itself or by can be requested by sending a mail at `abhishekbisht@gmail.com`.
- `config.ini` is needed to provide the directory paths such as those of datasets.
- `fusion_training_hyparam.json` is needed to supply the aprropriate hyperparameters for training.

**Acknowledgements**

This work uses the following external resources.
- `haarcascade_frontalface_alt2.xml` taken from [opencv](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml).
- `shape_predictor_5_face_landmarks.dat` taken from the repository [ageitgey/face_recognition_models](https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_5_face_landmarks.dat) by [Adam Geitgey](https://github.com/ageitgey).
- Datasets
    - [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)
    - [RAVDESS](https://zenodo.org/record/1188976#.YUYLUXUzaV5)
- Several research papers were also referred and the authors have been appropriately credited in the research paper associated with this work.
