# 🧠 Digit Recognizer — AI Neural Network Demo

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)

## 📌 What it does

This project is a full-stack, AI-powered web application that recognizes hand-drawn digits in real-time. 

Users draw any number from **0 to 9** on a sleek, glassmorphic digital canvas. The app immediately captures the drawing, normalizes it using advanced mathematical transformations (Center of Mass translations) to perfectly match the official MNIST dataset format, and passes it to a custom **PyTorch Neural Network**. The model then predicts the digit and displays the result along with its confidence percentage.

## ✨ Features

- **Modern Glassmorphism UI**: A highly responsive, single-page dark mode design with sleek gradient accents and micro-animations.
- **Flawless Mobile Support**: The drawing canvas and app scale intelligently. It prevents annoying scroll-overflows but elegantly unlocks smooth scrolling when predictions require more space.
- **True MNIST Preprocessing**: Real-world hand drawings are often messy. The backend runs complex Python `scipy.ndimage` center-of-mass spatial transformations and peak-brightness normalization to guarantee high model accuracy, fixing common "thin stroke" issues.
- **Custom Trained Neural Network**: Utilizes a 4-layer Multi-Layer Perceptron (MLP) trained to 98%+ accuracy on the classic MNIST dataset.

---

## 🛠️ Technology Stack

**Frontend:**
- **React.js** (via Vite)
- **Vanilla CSS** (Custom CSS variables, Glassmorphism, Responsive `vh/vw` constraints)

**Backend:**
- **Python / Flask** (REST API)
- **PyTorch** (Neural Network Inference)
- **NumPy & SciPy** (Advanced image matrix transformations)
- **Pillow (PIL)** (Image decoding)

---

## 🚀 Local Development Setup

To run this project locally, you will need two terminal windows: one for the Python backend and one for the React frontend.

### 1. Backend Setup (Flask API)
Open a terminal and navigate to the `api` folder:

```bash
cd api

# (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies (Downloads the lightweight CPU-only PyTorch)
pip install -r requirements.txt

# Start the Flask server
python app.py
```
*The API will start running on `http://localhost:5000`.*

### 2. Frontend Setup (React/Vite)
Open a second terminal and navigate to the `web` folder:

```bash
cd web

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*The frontend will start running on `http://localhost:5173`.*

---

## 📂 Project Structure

```text
├── api/
│   ├── app.py                     # Flask server and image preprocessing logic
│   ├── digit_recognizer_model.pth # Saved PyTorch neural network weights
│   └── requirements.txt           # Python dependencies (CPU-optimized)
├── web/
│   ├── src/                       # React frontend source code
│   │   ├── App.jsx                # Main canvas and prediction logic
│   │   ├── App.css                # Component-specific styles
│   │   └── index.css              # Global tokens and animations
│   ├── index.html                 # HTML Entry point
│   └── vite.config.js             # Vite configuration
└── digit_recognizer_training.ipynb # Jupyter Notebook containing the training loop
```

---

## 🌐 Deployment Architecture

This project requires a **Two-Part Deployment** because the lightweight frontend and heavy machine learning backend serve different purposes.

### Deploying the Backend
Due to the sheer size of the PyTorch package (~800MB), standard serverless platforms (like Vercel) often throw a "Max Size Exceeded" error. 
* **Recommendation**: Deploy the `api/` folder as a Web Service on **Render.com** or **Railway.app** using the included `requirements.txt`. These platforms support larger container sizes perfect for Machine Learning models.

### Deploying the Frontend
1. Once your API is live (e.g. `https://my-digit-api.onrender.com`), open `web/src/App.jsx`.
2. Change the `API_URL` constant from `http://localhost:5000` to your new live backend URL.
3. Push to GitHub.
4. Deploy the `web/` folder seamlessly to **Vercel** or **Netlify**.

---

## 📜 License
This project is open-source and available under the MIT License. Feel free to fork, modify, and learn from it!
