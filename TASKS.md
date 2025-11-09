# Project Tasks and Work Split

## Project Breakdown

The project can be broken down into the following key areas:

1. **Data Preparation:** Parsing annotation files, creating a custom dataset, and setting up data loaders.
2. **Model Development:** Defining the neural network architecture and implementing the training process.
3. **Prediction & Evaluation:** Using the trained model for inference, evaluating its performance, and potentially building a user interface.

### Taskset 1: Data Loader and Preprocessor

* **Task 1: Parse `.mat` files:**
  * **File:** `src/data_preprocessing.py`
  * **Description:** Write a script to read `cars_train_annos.mat` and `cars_meta.mat` (located in `data/dataset/car_devkit/devkit/`). This script should extract image filenames, bounding box coordinates, and corresponding class labels. It should also establish a mapping between class IDs and human-readable car brand names.

* **Task 2: Create a Custom Dataset Class:**
  * **File:** `src/dataset.py`
  * **Description:** Implement a custom dataset class (e.g., `torch.utils.data.Dataset` for PyTorch or `tf.data.Dataset` for TensorFlow). This class will handle loading individual images, applying necessary transformations (e.g., resizing, cropping images based on bounding box information), and performing data augmentation (e.g., random flips, rotations, color jittering).

* **Task 3: Prepare Data Loaders:**
  * **File:** `src/data_preprocessing.py` (or `src/dataset.py`)
  * **Description:** Develop functions to create data loaders (e.g., `torch.utils.data.DataLoader` or `tf.data.Dataset` pipelines) for both the training and validation sets. These loaders will efficiently feed batches of preprocessed data to the model during training.

### Taskset 2: Model Architecture and Training

* **Task 1: Define the Model Architecture:**
  * **File:** `src/model.py`
  * **Description:** Define the convolutional neural network (CNN) architecture. It is highly recommended to leverage a pre-trained model (e.g., ResNet, VGG, EfficientNet from `torchvision.models` or `tf.keras.applications`) and adapt it for car brand classification by replacing or modifying the final classification layer to output 196 classes.

* **Task 2: Implement the Training Loop:**
  * **File:** `src/train.py`
  * **Description:** Write the main script for training the model. This involves setting up the optimizer (e.g., Adam, SGD), defining the loss function (e.g., Cross-Entropy Loss), and implementing the training loop that iterates through epochs, performs forward and backward passes, and updates model weights.

* **Task 3: Experiment with Hyperparameters:**
  * **File:** `src/train.py` (or a separate configuration file)
  * **Description:** Conduct experiments with various hyperparameters such as learning rate, batch size, number of epochs, and different optimizers to optimize model performance. Track and log training progress (loss, accuracy).

### Taskset 3: Prediction, Evaluation, and User Interface

* **Task 1: Implement Prediction Script:**
  * **File:** `src/predict.py`
  * **Description:** Create a script that can load a trained model, take a new car image as input, preprocess it, and output the predicted car brand. This script should be robust and handle various input image formats.

* **Task 2: Model Evaluation:**
  * **File:** `src/predict.py` (or a separate `src/evaluate.py`)
  * **Description:** Develop functionality to evaluate the trained model's performance on the test dataset (`data/dataset/cars_test`). This includes calculating metrics such as accuracy, precision, recall, and F1-score. The evaluation should use the `cars_test_annos.mat` file for ground truth (though it lacks class labels, it provides image filenames for matching).

* **Task 3 (Optional but Recommended): Build a Simple User Interface:**
  * **File:** `src/app.py`
  * **Description:** Create a basic web application (e.g., using Flask, Streamlit, or Gradio) that allows users to upload an image of a car and receive the model's predicted car brand. This will demonstrate the practical application of the trained model.
