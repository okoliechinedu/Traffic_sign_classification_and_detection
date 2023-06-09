# Traffic Sign Recognition using Convolutional Neural Network (CNN)

This repository contains code for a Traffic Sign Recognition system using a Convolutional Neural Network (CNN). The system can classify traffic signs into different categories based on input images.

The code is written in Python and utilizes the Keras library with TensorFlow backend for building and training the CNN model. The model is trained on a dataset of traffic sign images, and the trained model is then used for real-time classification of traffic signs using a webcam.

## Key Features

- Imports and preprocesses traffic sign images from a given dataset
- Splits the data into training, testing, and validation sets
- Performs data augmentation to increase the diversity of the training data
- Builds a CNN model with multiple convolutional and pooling layers
- Trains the model on the training data with validation during the training process
- Evaluates the model performance on the test set and displays accuracy
- Saves the trained model for future use
- Uses the trained model for real-time traffic sign classification using a webcam

## Requirements

- Python 3.x
- OpenCV
- Keras with TensorFlow backend
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Usage

1. Clone the repository:

```
git clone https://github.com/your_username/traffic-sign-recognition.git
```

2. Install the required dependencies.

3. Prepare your traffic sign dataset by organizing the images in a directory structure where each subdirectory represents a class/category of traffic signs.

4. Update the path, batch size, and other parameters in the training code (`train.py`) according to your dataset and preferences.

5. Run the training code to train the CNN model:

```
python train.py
```

6. After training, a trained model file (`model_trained.p`) will be saved.

7. Connect a webcam to your computer.

8. Update the model file path in the testing code (`test.py`) to load the trained model.

9. Run the testing code to start the traffic sign recognition system:

```
python test.py
```

10. The webcam feed will open, and the system will display the detected traffic sign class and its probability in real-time.


## Acknowledgments

- The dataset used in this project is based on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
- The code is inspired by various tutorials and resources on traffic sign recognition and CNNs.

Feel free to contribute to this repository by creating pull requests for improvements or bug fixes. If you have any questions or suggestions, please open an issue.
