# diabetic-toungue-classifier-with-cnn
# 🩺 Diabetic Tongue Classification using Deep Learning

This project focuses on the binary classification of tongue images to detect diabetes using a Convolutional Neural Network (CNN) built with TensorFlow. The model learns to distinguish between diabetic and non-diabetic tongue features, enabling a non-invasive and efficient screening tool.

---

## 📌 Key Features

- ✅ **High Accuracy**: Achieved **97% accuracy** on the validation and test datasets.
- 🧠 **Model Architecture**: Custom CNN with multiple convolutional and pooling layers.
- 🔄 **Data Augmentation**: Improved generalization with:
  - Random rotation
  - Horizontal & vertical flips
  - Zoom and brightness variation
- 🛡️ **Regularization Techniques**:
  - `Dropout` layers to reduce overfitting
  - `L2 Regularization` to penalize complex weights
- 🖥️ **Web App**: Interactive frontend built using **Streamlit** for easy demo and real-time inference
- ⚖️ **Model Comparison**:
  - Outperforms fine-tuned **Vision Transformers (ViT)** and **EfficientNetB0**
  - Offers **faster inference** and **higher accuracy** with fewer parameters

---

## 🧪 Dataset

- **Input**: Preprocessed tongue images
- **Labels**: `0` - Non-Diabetic, `1` - Diabetic
- **Source**: Medical imaging sources (curated dataset)
- **Preprocessing**: Resizing, normalization, augmentation

---

## 🧰 Tech Stack

- `Python`, `TensorFlow / Keras`
- `OpenCV` for image handling
- `Streamlit` for frontend
- `Matplotlib`, `Seaborn` for plotting

---

## 📈 Results

| Model              | Accuracy | Inference Time | Remarks                     |
|--------------------|----------|----------------|-----------------------------|
| Custom CNN         | 97%      | Fast           | Best overall performance    |
| EfficientNetB0     | 99%      | Moderate       | no generalization           |
| ViT (fine-tuned)   | 99%      | Slow           | no generalization           |

---

## 📂 Project Structure


[watch how it works <- by clicking here](https://drive.google.com/file/d/11X6KetzA6FOzUv0yEnJBqKKLc7yliOV9/view?usp=sharing)
