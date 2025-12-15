# RF Signal Analyzer

A machine learning-powered web application for analyzing and classifying RF (Radio Frequency) spectrogram images. Upload a spectrogram image and instantly receive signal classification, confidence scores, and extracted band parameters.

## Features

- **Signal Classification**: Identifies 15+ RF signal types including:
  - RS41 Radiosonde
  - Airband, AIS, Bluetooth
  - Cellular, WiFi, LoRa, Z-Wave
  - FM, Digital Audio Broadcasting
  - SSTV, VOR, Packet, Remote Keyless Entry
  - And more...

- **Parameter Extraction**: Automatically extracts:
  - Normalized center frequency
  - Normalized bandwidth
  - SNR-like quality score
  - Band position (start/end columns)

- **User-Friendly Interface**: Built with Streamlit for easy uploads and visualization

- **JSON Export**: Download analysis results in JSON format for further processing

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone or download this repository

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Steps:
1. Upload a spectrogram image (PNG or JPG)
2. The model will classify the RF signal and extract parameters
3. View the prediction confidence and band parameters
4. Optionally download the analysis results as JSON

## Project Structure

```
├── app.py                      # Main Streamlit application
├── rf_utils.py                 # Signal analysis utility functions
├── rf_classifier.keras         # Pre-trained TensorFlow model
├── kept_classes.json           # List of signal class names
├── class_names.json            # Additional class information
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## File Descriptions

- **app.py**: Web interface for uploading spectrograms and displaying results
- **rf_utils.py**: Core analysis functions for signal classification and parameter extraction
  - `load_spectrogram_gray()`: Load and normalize grayscale spectrogram images
  - `extract_frequency_band_params()`: Extract frequency band characteristics
  - `analyze_signal()`: Perform signal classification and analysis
- **rf_classifier.keras**: Pre-trained Keras model for RF signal classification
- **kept_classes.json**: JSON array containing all supported signal class labels

## Model Details

The classifier uses a pre-trained deep learning model (`rf_classifier.keras`) that:
- Accepts 224×224 pixel RGB spectrogram images
- Outputs predictions across 15+ RF signal classes
- Provides confidence scores for each prediction

## Output Format

The JSON export includes:
```json
{
  "predicted_class": "signal_type",
  "confidence": 0.95,
  "band_params": {
    "center_freq_norm": 0.45,
    "bandwidth_norm": 0.12,
    "snr_like": 0.35,
    "start_col": 100,
    "end_col": 127
  }
}
```

## Troubleshooting

- **Model not found**: Ensure `rf_classifier.keras` is in the same directory as `app.py`
- **Module not found**: Run `pip install -r requirements.txt` to install all dependencies
- **Image upload issues**: Ensure images are PNG or JPG format and clearly show the spectrogram

## Model Training

The model was trained using a Jupyter notebook in Google Colab. You can view the training process here:
[RF Signal Classifier Training Notebook](https://colab.research.google.com/drive/1T1V7_8ccFoCodbhAFJzHp4zmS3Z0Aw8k?usp=sharing)


