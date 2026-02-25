# 🧠 CIFAR-10 Image Classification using CNN (AlexNet Inspired)

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) model inspired by the AlexNet architecture to perform image classification on the CIFAR-10 dataset.

The model is built and trained using TensorFlow and Keras in Google Colab.

This project demonstrates:

- Image preprocessing
- CNN model building
- Model training and validation
- Performance evaluation
- Prediction visualization

---

## 📂 Dataset

The model is trained on the **CIFAR-10 dataset**, which contains:

- 60,000 color images (32x32 pixels)
- 10 different classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

Dataset split:
- 50,000 training images
- 10,000 testing images

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Normalization (pixel values scaled to 0–1 range)
- Data Augmentation:
  - Random horizontal flip
  - Random translation
  - Random rotation

This improves model generalization and reduces overfitting.

---

## 🏗 Model Architecture

The model consists of:

- Convolutional Layers (Conv2D)
- ReLU activation function
- MaxPooling layers
- Flatten layer
- Fully Connected (Dense) layers
- Softmax output layer (10 classes)

Optimizer used:
- Adam

Loss function:
- Sparse Categorical Crossentropy

Training:
- 10 epochs
- Validation performed on test dataset

---

## 📊 Model Evaluation

The model is evaluated on the CIFAR-10 test dataset.

Outputs:
- Test accuracy
- Loss value

The notebook also visualizes:
- Predicted labels vs True labels
- Sample test image predictions

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab

---

## 🚀 How To Run

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/repository-name.git
   ```

2. Open the notebook in Google Colab.

3. Run all cells sequentially to train and evaluate the model.

---

## 📈 Learning Outcomes

Through this project, the following concepts were implemented and understood:

- Convolutional Neural Networks (CNN)
- Image normalization
- Data augmentation techniques
- Model compilation and training
- Accuracy and loss evaluation
- Prediction and visualization

---

## 🔮 Future Improvements

- Implement full AlexNet architecture
- Add Batch Normalization
- Increase number of epochs
- Hyperparameter tuning
- Compare performance with modern architectures (ResNet, VGG)

---

## 👨‍💻 Author

Ashutosh Kumar  
Deep Learning & AI Enthusiast
