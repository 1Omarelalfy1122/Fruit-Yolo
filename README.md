# 🍊 Orange Quality Detection (Good vs Defected)

This project uses **YOLO11s** (You Only Look Once) for detecting and classifying oranges into two categories:
- ✅ Good Orange  
- ❌ Defected Orange  

The model was trained on a custom dataset and can be used to automatically evaluate the quality of oranges from images.

---

## 📂 Project Structure
Fruit-Yolo/
│── data.yaml # Dataset configuration
│── requirements.txt # Dependencies
│── runs/ # YOLO training results (auto-generated)
│── images/ # Test images
│── custom_data/ # Original dataset (images + labels)
│── train.py # Training script (YOLO command)
│── test.py # Inference script for testing images



---

## 🚀 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/1Omarelalfy1122/Fruit-Yolo.git
cd Fruit-Yolo

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt


