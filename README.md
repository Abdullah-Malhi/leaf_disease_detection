# leaf disease detection and classification
## Overview
This notebook demonstrates the complete workflow for training a convolutional neural network using the DenseNet201 architecture to classify 67 classes of plant diseases from images. It includes steps for setting parameters, loading datasets, data preprocessing and augmentation, model definition using the functional API, model compilation, training with custom callbacks, and saving the trained model and training history. The notebook also provides a function to display random images from each class to visualize the dataset.
## Dataset
We combined multiple datasets from Kaggle to create our own comprehensive dataset, which is available on Kaggle: [Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/abdullahmalhi/leaf-diseases-extended). The model is trained using TensorFlow/Keras and evaluated on this dataset with the notebook created on Kaggle. The dataset used for training and evaluation consists of images collected from various sources, encompassing a wide range of leaf diseases and healthy states. The dataset is organized into training, validation, and test sets, allowing for rigorous model evaluation.

![image](https://github.com/user-attachments/assets/bfb4cbb0-3548-4f21-b2b4-35e66e6adb8e)

## Model
Our trained model is available on Kaggle: [Leaf Disease Detection Model](https://www.kaggle.com/models/isramansoor9/leaf_disease_detection_model)

![image](https://github.com/user-attachments/assets/f28750ed-9e9e-4745-b5e9-0792b91ec706)

![image](https://github.com/user-attachments/assets/c4998e31-9760-48e9-9af6-1412906c2ad6)

## Features
* **DenseNet201 Architecture**: Utilizes the DenseNet201 architecture for deep learning.
* **Data Augmentation**: Implements data augmentation techniques to enhance model generalization.
* **Custom Callbacks**: Employs custom callbacks to monitor training progress and improve performance.
* **Model Visualization**: Provides functions to visualize random images from each class to understand the dataset better.
* **Training History**: Saves training history for future analysis and model improvement.
## Usage
1. **Clone the Repository**: Clone this repository to your local machine using git clone <repository-url>.
2. **Install Dependencies**: Install the required dependencies using pip install numpy pandas tensorflow matplotlib seaborn.
3. **Run the Notebook**: Open the notebook and run each cell sequentially to train the model.
    * If you prefer to use the pretrained model, download the pretrained model and model history from here.
    * Update the notebook to load the pretrained model and model history by changing the links in the training section to the specific links of the pretrained model and model history.
## Results
The trained model achieves high overall test accuracy of **97.03%** in classifying 67 different leaf diseases, demonstrating the effectiveness of using DenseNet201 for this task. The results, including accuracy and loss curves, are saved and can be visualized for further analysis.

![image](https://github.com/user-attachments/assets/e62718d2-a103-45c5-abfd-1b1822ecf32f)

![image](https://github.com/user-attachments/assets/b9c94c66-fe17-49c4-b189-5b03be4e474b)

## Confusion Matrix
![image](https://github.com/user-attachments/assets/eaab22bb-d7c6-40b9-b7a6-bb89b7372e18)

![image](https://github.com/user-attachments/assets/e15e6b7d-0841-452b-8130-86115d6b28e1)

![image](https://github.com/user-attachments/assets/fa44e4e8-940d-4769-ab46-66cf459a3245)


## Contributing
We welcome contributions! Please fork this repository and submit pull requests for any enhancements or bug fixes.
