# Image Classification with TensorFlow

## Project Overview
In this project aim to build and train a convolutional neural network (CNN) using TensorFlow and Keras. The dataset of choice is the Intel Image Classification dataset, a rich collection of approximately 25,000 images categorized into six classes. The model will be trained on a subset of this dataset and also validated on a separate testing set as well as on real-world pictures. The goal is to create a robust classifier that can accurately identify the category of a given image, which has numerous practical applications in various fields such as automated tagging, content filtering, and more.


## Dataset
The dataset used for this proejct is **Intel Image Classification**, which is a collection of natural scene images that are divided into six categories: **buildings**, **forest**, **glacier**, **mountain**, **sea**, and **street**. The dataset contains around 25,000 images of size 150x150 pixels. The dataset is catagorized into two parts, the first part is gonna be used for traning the model and the other one be used to testing the model accurcy (seg_train, seg_test).

The dataset is available on Kaggle at https://www.kaggle.com/datasets/puneet6060/intel-image-classification/code.


## Model Architecture
The model is a sequential CNN with the following layers:
- Conv2D with 32 filters, kernel size of 3, and ReLU activation
- MaxPooling2D with pool size of 2
- Conv2D with 64 filters, kernel size of 3, and ReLU activation
- MaxPooling2D with pool size of 2
- Conv2D with 128 filters, kernel size of 3, and ReLU activation
- MaxPooling2D with pool size of 2
- Flatten layer
- Dense layer with 256 units and ReLU activation
- Output Dense layer with 6 units and softmax activation

## Training
The model is compiled with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the metric. It is trained for 2 epochs using the training and validation generators.

## Evaluation
The model's performance is evaluated using a test generator, which processes images from the test set.

## Custom Predictions
The project also includes code to load and preprocess custom images from Google Drive, predict their classes using the trained model, and display the images with their predicted labels.

## How to Use
1. Mount your Google Drive to access the dataset.
2. Define the image size, batch size, and class names.
3. Create data generators for training, validation, and testing.
4. Define and compile the model.
5. Train the model using the `fit` method.
6. Evaluate the model's performance.
7. Load and preprocess custom images for predictions.

## Requirements
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL
- Google Colab

## Acknowledgments
This project was completed as part of the Advanced AI course at [Your University/Institution].

## License
[Choose an appropriate license for your project, such as MIT, GPL, etc.]

