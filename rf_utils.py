import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

def load_spectrogram_gray(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype("float32") / 255.0
    return img_gray

def extract_frequency_band_params(img_gray, energy_threshold_ratio=0.3):
    h, w = img_gray.shape
    col_energy = img_gray.mean(axis=0)
    max_energy = col_energy.max()
    if max_energy <= 0:
        return None
    threshold = max_energy * energy_threshold_ratio
    active = col_energy > threshold
    if not np.any(active):
        return None
    indices = np.where(active)[0]
    start = int(indices[0])
    end = int(indices[-1])
    center_x = (start + end) / 2.0
    band_width = float(end - start + 1)
    center_freq_norm = center_x / float(w)
    bandwidth_norm = band_width / float(w)
    band_energy = col_energy[start:end+1].mean()
    bg_mask = np.ones_like(col_energy, dtype=bool)
    bg_mask[start:end+1] = False
    bg_energy = col_energy[bg_mask].mean() if np.any(bg_mask) else 0.0
    snr_like = float(band_energy - bg_energy)
    return {
        "center_freq_norm": float(center_freq_norm),
        "bandwidth_norm": float(bandwidth_norm),
        "snr_like": float(snr_like),
        "start_col": start,
        "end_col": end
    }
    
def analyze_signal(model, class_names, img_path, image_size):
    img = load_img(img_path, target_size=image_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    cls_idx = int(np.argmax(preds))
    cls_label = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
    cls_conf = float(preds[cls_idx])
    img_gray = load_spectrogram_gray(img_path)
    band_params = extract_frequency_band_params(img_gray)
    return {
        "predicted_class": cls_label,
        "confidence": cls_conf,
        "band_params": band_params
    }

