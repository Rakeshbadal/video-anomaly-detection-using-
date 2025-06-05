# video-anomaly-detection


# Video Anomaly Detection Pipeline (ConvLSTM3D + ESRGAN + WGAN-GP)

## Folder Structure

```
project_root/
├── dataset/
│   ├── training/
│   ├── testing/
│   └── testing_labels.npy
├── models/
│   ├── convlstm3d.py
│   ├── esrgan.py
│   ├── wgan_gp.py
├── utils.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## requirements.txt
```
torch
numpy
scikit-learn
Pillow
matplotlib
tqdm
torchvision
```

---

## README.md
```markdown
# Video Anomaly Detection (ConvLSTM3D + ESRGAN + WGAN-GP)

This project implements a video anomaly detection pipeline combining:
- **ConvLSTM3D** for temporal sequence modeling
- **ESRGAN** for spatial refinement
- **WGAN-GP** for adversarial training

## 📦 Structure
- `models/`: Contains model architectures
- `utils.py`: Dataset loader
- `train.py`: Model training script
- `test.py`: Anomaly scoring + AUC evaluation

## 🧪 Dataset Format
Images must be stored as:
```
dataset/training/<scene_name>/frameXXXX.jpg
dataset/testing/<scene_name>/frameXXXX.jpg
```

You must also provide ground-truth anomaly labels for test sequences:
```python
# Save array like [0, 0, 1, 0, ...]
np.save("dataset/testing_labels.npy", np.array([...]))
```

## 🚀 Train
```bash
python train.py
```

## 🔍 Test
```bash
python test.py
```

## 📈 Output
Prints normalized anomaly scores and AUC score based on L1 difference between predicted and real frames.

---

## 🛠 Tips
- Resize input frames to 64x64
- Tune ConvLSTM and ESRGAN depth for better results
- Consider using a better perceptual loss like LPIPS for advanced scoring
```

Let me know if you want all the `*.py` files to be copied here in full too!
