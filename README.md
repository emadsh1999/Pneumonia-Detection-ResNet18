**Pneumonia Detection from Chest X-Rays using ResNet18
**📝 Project Overview
This repository contains a Deep Learning pipeline designed to automate the detection of Pneumonia from chest X-ray images. By leveraging Transfer Learning with the ResNet18 architecture, the model aims to provide a rapid diagnostic tool to support clinical decision-making.

📊 Dataset
The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle. It consists of validated pediatric chest X-ray images categorized into two classes: Normal and Pneumonia.

Exploratory Data Analysis
Below are samples of the X-ray images used for training, showcasing the visual differences between healthy lungs and those affected by pneumonia.

X-ray Samples

⚙️ Methodology
Architecture: ResNet18 (Residual Network).
Transfer Learning: Initialized with pre-trained ImageNet weights to improve feature extraction on medical imagery.
Image Preprocessing:
Resizing to 224x224 pixels.
Grayscale-to-RGB conversion (3-channel compatibility).
Normalization using ImageNet statistics.
Optimization:
Optimizer: Adam (LR = 1e-4)
Loss Function: Cross-Entropy Loss
Training Duration: 30 Epochs
📈 Results
Training Performance
The model achieved steady convergence over the 30-epoch training cycle.

Training Loss

Model Evaluation
Evaluation was performed on the test set, analyzing precision, recall, and overall accuracy.

Confusion Matrix
Confusion Matrix

Inference Samples
Visual demonstration of the model predicting on unseen test data with associated confidence scores.

Prediction Samples

🚀 How to Use
Environment Setup: Install dependencies using pip install -r requirements.txt.
Model Loading: Load the trained weights using PyTorch:
model.load_state_dict(torch.load('models/pneumonia_resnet18.pth'))
Run Inference: Use the provided notebook in the notebooks/ folder to run predictions on your own X-ray images.
🛠️ Future Improvements
Class Balancing: Implementing Weighted Cross-Entropy Loss to handle the higher frequency of Pneumonia cases.
Data Augmentation: Applying rotations and flips to the training set to improve generalization.
Advanced Architectures: Testing ResNet50 or DenseNet121 for potentially higher accuracy.
