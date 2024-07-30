# leaf disease classificaton and segmentation

## Table of Contents
1. [Classification](#classification)
   - [Overview](#classification-overview)
   - [Dataset](#classification-dataset)
   - [Model](#classification-model)
   - [Features](#classification-features)
   - [Results](#classification-results)
   - [Confusion Matrix](#classification-confusion-matrix)
2. [Segmentation](#segmentation)
   - [Overview](#segmentation-overview)
   - [Dataset](#segmentation-dataset)
   - [Model](#segmentation-model)
   - [Features](#segmentation-features)
   - [Results](#segmentation-results)
3. [Usage](#usage)
4. [Contributing](#contributing)

# Classification
## Overview <a id="classification-overview"></a>
This notebook demonstrates the complete workflow for training a convolutional neural network using the DenseNet201 architecture to classify 67 classes of plant diseases from images. It includes steps for setting parameters, loading datasets, data preprocessing and augmentation, model definition using the functional API, model compilation, training with custom callbacks, and saving the trained model and training history. The notebook also provides a function to display random images from each class to visualize the dataset.

## Dataset <a id="classification-dataset"></a>
We combined multiple datasets from Kaggle to create our own comprehensive dataset, which is available on Kaggle: [Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/abdullahmalhi/leaf-diseases-extended). The model is trained using TensorFlow/Keras and evaluated on this dataset with the notebook created on Kaggle. The dataset used for training and evaluation consists of images collected from various sources, encompassing a wide range of leaf diseases and healthy states. The dataset is organized into training, validation, and test sets, allowing for rigorous model evaluation.

![image](https://github.com/user-attachments/assets/bfb4cbb0-3548-4f21-b2b4-35e66e6adb8e)

## Model <a id="classification-model"></a>
Our trained model is available on Kaggle: [Leaf Disease Detection Model](https://www.kaggle.com/models/isramansoor9/leaf_disease_detection_model)

![image](https://github.com/user-attachments/assets/f28750ed-9e9e-4745-b5e9-0792b91ec706)

![image](https://github.com/user-attachments/assets/c4998e31-9760-48e9-9af6-1412906c2ad6)

## Features <a id="classification-features"></a>
* **DenseNet201 Architecture**: Utilizes the DenseNet201 architecture for deep learning.
* **Data Augmentation**: Implements data augmentation techniques to enhance model generalization.
* **Custom Callbacks**: Employs custom callbacks to monitor training progress and improve performance.
* **Model Visualization**: Provides functions to visualize random images from each class to understand the dataset better.
* **Training History**: Saves training history for future analysis and model improvement.
      
## Results <a id="classification-results"></a>
The trained model achieves high overall test accuracy of **97.03%** in classifying 67 different leaf diseases, demonstrating the effectiveness of using DenseNet201 for this task. The results, including accuracy and loss curves, are saved and can be visualized for further analysis.

![image](https://github.com/user-attachments/assets/e62718d2-a103-45c5-abfd-1b1822ecf32f)

![image](https://github.com/user-attachments/assets/b9c94c66-fe17-49c4-b189-5b03be4e474b)

## Confusion Matrix <a id="classification-confusion-matrix"></a>

![image](https://github.com/user-attachments/assets/e15e6b7d-0841-452b-8130-86115d6b28e1)

![image](https://github.com/user-attachments/assets/fa44e4e8-940d-4769-ab46-66cf459a3245)

# Segmentation
## Overview <a id="segmentation-overview"></a>
This notebook demonstrates the complete workflow for training and evaluating a DeepLabV3+ model with a ResNet101 backbone for semantic segmentation of leaf diseases. The segmentation process is tailored to classify and segment various leaf diseases from images.

## Dataset <a id="segmentation-dataset"></a>
The segmentation aspect of this project utilizes deep learning to identify and segment leaf diseases. The model is trained and tested on the [Leaf Disease Segmentation Dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset) which contains annotated images for segmentation tasks. To apply and evaluate the model on a wider variety of leaf diseases, we use our [Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/abdullahmalhi/leaf-diseases-extended). This broader dataset helps in assessing the model’s performance and robustness in segmenting different types of leaf diseases.

## Model <a id="segmentation-model"></a>
Our trained model deeplabv3plus_resnet101 is available on Kaggle: [Leaf Disease Segmentation Model](https://www.kaggle.com/models/abdullahmalhi/deeplabv3plus_resnet101)

![image](https://github.com/user-attachments/assets/87e2c874-4efa-4e3d-905f-2b8f427bab79)

## Features <a id="segmentation-features"></a>
* **Data Loading and Processing**: Efficiently loads and processes image and mask data, resizing them to a consistent shape (256x256 pixels) for segmentation tasks.
* **Data Augmentation**: Implements augmentation techniques such as horizontal and vertical flips, rotation, and shifts to enhance the training dataset and improve model robustness.
* **Custom Convolutional Block**: Defines a ConvBlock layer for use in the model, incorporating convolution, batch normalization, and activation functions.
* **Atrous Spatial Pyramid Pooling (ASPP)**: Integrates ASPP to capture multi-scale contextual information from input images, improving segmentation accuracy.
* **DeepLabV3+ Model Architecture**: Constructs a DeepLabV3+ model using a ResNet101 backbone, designed for precise segmentation of leaf diseases.
* **Model Training and Evaluation**: Trains the model using a binary cross-entropy loss function and Adam optimizer, with checkpoints and progress visualization for monitoring.
* **Visualization**: Provides functions to visualize model predictions, including overlays of predicted masks on input images, and evaluates performance on the Leaf Diseases Extended Dataset.
* **Custom Callback**: Implements a custom callback to show model predictions at the end of certain epochs during training.

## Results <a id="segmentation-results"></a>

On given dataset

![image](https://github.com/user-attachments/assets/081c6049-4379-4983-abbd-a4c6820330f5)

On our dataset

![image](https://github.com/user-attachments/assets/81a5d939-fdb7-4790-aecd-ea2c186a75fa)

  
## Usage
1. **Clone the Repository**: Clone this repository to your local machine using git clone <repository-url>.
2. **Install Dependencies**: Install the required dependencies using pip install numpy pandas tensorflow matplotlib seaborn tqdm pillow tf_explain.
3. **Run the Notebook**: Open the notebook and run each cell sequentially to train the model.
    * If you prefer to use the pretrained model, download the pretrained model and model history from here.
    * Update the notebook to load the pretrained model and model history by changing the links in the training section to the specific links of the pretrained model and model history.

## Contributing
We welcome contributions! Please fork this repository and submit pull requests for any enhancements or bug fixes.
