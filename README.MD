# CNN Signature Analysis POC
Deep Learning Proof of Concept for Analysing Authenticity of Signatures using Tensorflow CNN (https://en.wikipedia.org/wiki/Convolutional_neural_network). 

## Getting Started

### Installation
To install required packages, run `pip install -r requirements.txt`.

### Training / Testing
* Data will be read from the `dev_data` folder. Place images within `dev_data` (under `train/test`, `forge/real`) to train & test the model based on input segregated data.
* To train and test the model, run `python train_test.py`. This will generate a `model` folder storing the model information
* To increase the number of training steps, edit the `N_STEPS` variable under `train_test.py`.

### Prediction
* Data files will be read under `predict_data` folder.
* To run predictions on images under the above folder, run `python predict.py`. Predictions will show on stdout.

### Notes
* This POC mainly tests out a sample signatures for 2 people. Further upgrade possible, to segregate model folders, each customized to a particular person. Each model folder will be tagged with an ID, with binary classification of forge/real for a particular person. 

## Built With
* [Tensorflow](https://www.tensorflow.org/) - Open Source Machine / Deep Learning Platform
* [OpenCV](https://opencv.org/) - Open Source Computer Vision & Machine Learning Library
* [Python](https://www.python.org/downloads/release/python-362/) - Python 3.6.2