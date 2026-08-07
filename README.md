# 🫁 LungVision: Clinical AI Trustworthiness & Interpretability Audit

LungVision is a Streamlit-based web application designed to audit the trustworthiness and interpretability of a lung cancer detection model. It utilizes Explainable AI (XAI) techniques, specifically Grad-CAM, to visualize the regions of a biopsy slide that influence the model's predictions. 

The tool includes a **Clinical Noise Simulator** to apply H&E stain shift severities, mimicking different hospital staining protocols. By comparing the baseline model reasoning against stressed reasoning (with simulated clinical variance), researchers and clinicians can identify potential "shortcut learning"—where the model overfits to color variances rather than relying on underlying biological morphology.

## ✨ Features

- **Upload Biopsy Slides:** Upload histological images for analysis.
- **Explainable AI (Grad-CAM):** Visualizes pixel-level importance used by the model for its prediction.
- **Clinical Noise Simulation:** Adjusts image properties (brightness, contrast, saturation, hue) to mimic variations in H&E staining.
- **Reasoning Consistency Audit:** Side-by-side comparison of heatmaps to ensure the model focuses on cellular structures instead of color artifacts.

## 🛠️ Prerequisites

To run this application, you need Python installed and the following dependencies:

```bash
pip install -r requirements.txt
```

*(Note: Ensure you create a `requirements.txt` file containing dependencies like `streamlit`, `torch`, `torchvision`, `Pillow`, `numpy`, and `grad-cam`)*

## 🚀 Running the App

1. Ensure you have the model weights file `lung_cancer_xai_model.pth` in the root directory.
2. Run the Streamlit application:

```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: The main Streamlit application containing the UI and logic.
- `Lung_Cancer_detection_research.ipynb`: Jupyter Notebook containing the research, model training, and experimentation.
- `lung_cancer_xai_model.pth`: The trained PyTorch model weights.
- `Graphs/`: Directory containing generated plots and graphs from the research phase.

## ⚠️ Important Note on Large Files

The PyTorch model file (`lung_cancer_xai_model.pth`) is quite large (~94MB). If you plan to clone or fork this repository, ensure you have Git LFS (Large File Storage) installed if the model is tracked using it, or download the weights from the designated release page if hosted separately.

## Authors

**Nitin Kumar**
