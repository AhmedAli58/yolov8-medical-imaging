# YOLOv8 Medical Imaging

YOLOv8-based solution for medical image classification, detection and segmentation across three clinical domains.

---

## Results

| Task | Dataset | Model | Metric | Score |
|------|---------|-------|--------|-------|
| Classification | COVID-19 Chest X-rays | YOLOv8s-cls | Accuracy | 96.5% |
| Detection | RBC/WBC Blood Cells | YOLOv8s | mAP50 | 99.3% |
| Segmentation | Breast Ultrasound | YOLOv8s-seg | mAP50 | 72.3% |

---

## Features

- COVID-19 Classification — classifies chest X-rays as Normal, COVID-19, or Pneumonia
- Blood Cell Detection — detects and counts RBC and WBC with bounding boxes
- Breast Ultrasound Segmentation — segments normal, benign and malignant masses

---

## Tech Stack

- YOLOv8 (Ultralytics)
- PyTorch
- Streamlit
- OpenCV
- Python 3.12

---

## Datasets

- COVID-19: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
- Blood Cells: https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset
- Breast Ultrasound: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

---

## How to Run
```bash
git clone https://github.com/AhmedAli58/yolov8-medical-imaging.git
cd yolov8-medical-imaging
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure
```
yolov8-medical-imaging/
├── models/
│   ├── covid_cls.pt
│   ├── bloodcells_det.pt
│   └── breast_seg.pt
├── app.py
├── requirements.txt
└── README.md
```

