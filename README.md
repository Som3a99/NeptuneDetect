# 🌊 DeepScan: Underwater Object Detection System

![Underwater Detection](https://img.shields.io/badge/Underwater-Detection-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![Streamlit](https://img.shields.io/badge/Platform-Streamlit-red)

## 🎯 Overview
DeepScan is a cutting-edge underwater object detection system powered by YOLOv8 and advanced deep learning techniques. Designed for marine environments, it identifies and classifies objects with high accuracy. Built on the enhanced TrashCan V1.0 dataset, it is optimized for deployment on Remotely Operated Vehicles (ROVs) and offers both local and cloud-based detection options.

## ⭐ Key Features
- 🎥 **Real-time Detection**: Live object tracking via webcam
- 📸 **Static Analysis**: Precise image and video processing
- ☁️ **Cloud Platform**: Access from anywhere
- 🎚️ **Custom Settings**: Adjustable detection sensitivity
- 📊 **Smart Analytics**: Comprehensive visualization tools
- 💾 **Export Tools**: Flexible data extraction options

## 📚 Dataset Spotlight
Our enhanced TrashCan 1.0 dataset features:

### 📦 Base Dataset
- 🔢 7,212 annotated images from J-EDI
- 🏷️ Instance segmentation labels
- 🌊 Diverse underwater scenarios

### 🚀 Enhancements
- 🧹 Advanced data cleaning
- ✨ Albumentations augmentation
- 🎯 Refined annotations
- 🌊 Marine-optimized processing

## 💻 Installation

### 📋 Requirements
```bash
# 🛠️ Core Dependencies
streamlit==1.28.0
ultralytics==8.0.196
opencv-python==4.8.1.78
pillow==10.0.1
torch==2.1.0
numpy==1.26.0
matplotlib==3.8.0
python-dotenv==1.0.0
```

### 🔧 Setup Steps
1. 📥 **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/deepscan.git
   cd deepscan
   ```

2. 📦 **Install Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. ⚙️ **Configure Model**:
   ```bash
   mkdir -p weights/detection
   # 🏗️ Add YOLOv8 weights here
   ```

## 🚀 Deployment

### 💻 Local Setup (Full Version)
1. 🔑 Set environment:
   ```bash
   export IS_LOCAL=true
   ```

2. 🎯 Launch app:
   ```bash
   streamlit run app.py
   ```

3. 🌐 Visit: `http://localhost:8501`

### ☁️ Cloud Setup (Static Detection Only)
- 🌐 Visit: `http://localhost:8501`

## 🎮 Usage Guide
1. 🎯 **Model Settings**:
   - 🤖 Select model
   - 🎚️ Set confidence level
   
2. 📥 **Input Methods**:
   - 🖼️ Image upload
   - 🎥 Video upload
   - 📹 Webcam (local only)

3. 📊 **Results**:
   - 🎯 View detections
   - 📈 Check statistics
   - 💾 Save results

## 📂 Project Structure
```
deepscan/
├── 📱 app.py           # Main app
├── ⚙️ config.py        # Settings
├── 🛠️ utils.py         # Tools
├── 📋 requirements.txt # Packages
├── 📦 weights/         # Models
└── 📸 captures/        # Images
```

## 👨‍💻 Developer Guide

### 🔧 Dev Setup
1. 🏗️ Create environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # 🐧 Linux/Mac
   venv\Scripts\activate     # 🪟 Windows
   ```

2. 📚 Install dev tools:
   ```bash
   pip install -r requirements.txt
   ```

### 🤝 Contributing
1. 🔱 Fork repo
2. 🌿 Create branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. 💾 Commit:
   ```bash
   git commit -m "✨ Add new feature"
   ```
4. 🚀 Submit PR


## 📄 License
[📜 Your License]

## 🙏 Acknowledgments
- 📚 TrashCan 1.0 Team
- 🤖 Ultralytics
- 💻 Streamlit Community

## 📫 Contact
- 📧 Email: [Mohamed_EMohamed@outlook.com]
- 💬 GitHub Issues: [link]

---
<div align="center">
🌊 Made with 💙 for Ocean Conservation 🌊
</div>
