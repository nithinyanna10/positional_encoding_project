# 🔡 Learnable Positional Encoding Visualizer with Streamlit

This project demonstrates a learnable positional encoding mechanism in PyTorch, integrated with GloVe word embeddings. It allows users to input custom sentences and visualize how positional encodings interact with token embeddings.



## 🚀 Features

- 📥 Automatic download and extraction of **GloVe 6B 100d** embeddings
- 🧠 Learnable positional encoding using PyTorch
- 🌐 Real-time sentence input via Streamlit UI
- 🔥 Side-by-side heatmap comparison of:
  - Raw GloVe Embeddings
  - Positional Encodings
  - Combined GloVe + Positional Embeddings
- 📷 Option to download high-resolution `.png` and `.svg` visualizations
- 🧪 Toy classifier (optional extension) for sentiment prediction

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/nithinyanna10/positional_encoding_project.git
cd positional_encoding_project

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

- torch
- numpy
- matplotlib
- seaborn
- tqdm
- requests
- urllib3
- streamlit

## 🏃‍♂️ Run the App

```bash
streamlit run app.py
```

## 🧠 About Learnable Positional Encoding

Traditional transformers use fixed sinusoidal positional encodings. This project demonstrates how to **learn positional encodings** as trainable parameters, allowing the model to adapt to sequence-specific patterns.

## 📸 Screenshots

## 📝 Notes

- The `glove.6B.100d.txt` file (\~331MB) is **not included** in the repo.
- It will be downloaded automatically on first run (make sure you're connected to the internet).
- Make sure `.gitignore` excludes this file to avoid GitHub file size limits.

## ✨ Author

Nithin Reddy Yanna

---


