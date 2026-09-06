# TruthLens AI

## An Explainable Ensemble Framework for AI-Generated and Manipulated Image Detection

---

## 1. Problem Statement

The rapid advancement of generative AI models — Stable Diffusion, Midjourney, DALL-E, StyleGAN, and others — has made it trivially easy to create photorealistic synthetic images. These images are now indistinguishable from real photographs to the human eye, creating a critical trust problem in digital media.

**Core Question:**

> Given an uploaded image, is it **Real**, **AI-generated**, or **Manipulated/Deepfake**?

**And more importantly:**

> **Why** does the model think so?

TruthLens AI is a production-style detection platform that answers both questions using an ensemble of deep learning models, frequency-domain analysis, metadata forensics, and explainable AI techniques.

---

## 2. Project Goals

| Goal | Description |
|------|-------------|
| **Detection** | Classify images as Real, AI-Generated, or Manipulated with high confidence |
| **Explainability** | Provide Grad-CAM heatmaps showing which regions triggered the prediction |
| **Forensics** | Extract and analyze EXIF metadata, compression artifacts, and editing software signatures |
| **Robustness** | Evaluate prediction stability under compression, resizing, cropping, and noise |
| **Generalization** | Detect images from unseen generators (cross-generator evaluation) |

---

## 3. System Architecture

```
                         ┌──────────────────┐
                         │      USER        │
                         └────────┬─────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │    React Frontend      │
                     │  Upload + Dashboard    │
                     └───────────┬────────────┘
                                 │
                              REST API
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │       FastAPI Gateway        │
                  └──────────────┬───────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     ┌─────────────┐      ┌─────────────┐    ┌──────────────┐
     │ Auth Service│      │ Image Queue │    │ History DB   │
     └─────────────┘      └──────┬──────┘    └──────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  AI Detection Service  │
                    └───────────┬────────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
    ┌───────────┐        ┌──────────────┐       ┌──────────────┐
    │ CNN Model │        │ ViT Model    │       │ Frequency    │
    │EfficientNet│       │ Transformer  │       │ Analysis     │
    └─────┬─────┘        └──────┬───────┘       └──────┬───────┘
          │                     │                       │
          └─────────────────────┼───────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Ensemble Engine     │
                     └──────────┬──────────┘
                                │
                ┌───────────────┼────────────────┐
                ▼                                ▼
       ┌────────────────┐              ┌────────────────┐
       │ Explainability │              │ Metadata       │
       │ Grad-CAM       │              │ Forensics      │
       └────────┬───────┘              └───────┬────────┘
                │                              │
                └─────────────────┬────────────┘
                                  ▼
                     ┌──────────────────┐
                     │ Final AI Verdict │
                     └──────────────────┘
```

---

## 4. AIML Pipeline

### Stage 1: Image Input

```
Uploaded Image
      │
      ├── Validate Format (JPG, PNG, WEBP)
      ├── Check File Size (< 20MB)
      ├── Calculate SHA256 Hash
      └── Store Original
```

### Stage 2: Preprocessing Pipeline

```
Original Image
      │
      ├── RGB Version ───────→ CNN / ViT
      ├── Grayscale ─────────→ Texture Analysis
      ├── FFT Transform ─────→ Frequency Analysis
      └── Metadata ──────────→ Forensics
```

This makes the system **multimodal at the feature level** — each analysis branch receives the input representation it needs.

---

## 5. Model Architecture

### Model A: Custom CNN (Baseline)

The starting point — a from-scratch CNN to establish a genuine baseline. Implemented in Keras/TensorFlow (`notebooks/Baseline_Model_1.ipynb`). Inputs are channels-last `(32, 32, 3)`; `padding="same"` keeps the spatial size at 32×32, so after three 2×2 MaxPooling layers it is 4×4 and `Flatten` yields `128 × 4 × 4 = 2048` — the same 2048 → 256 → output projection as the original PyTorch-style spec. `notebooks/Improve_Baseline_2.ipynb` is the augmentation variant (horizontal flip added) — it did **not** improve the baseline (see Phase 1 note).

```
Input (32, 32, 3)
    ↓
Conv2D(32, 3, padding="same") → BatchNorm → ReLU → MaxPool(2)
    ↓
Conv2D(64, 3, padding="same") → BatchNorm → ReLU → MaxPool(2)
    ↓
Conv2D(128, 3, padding="same") → BatchNorm → ReLU → MaxPool(2)
    ↓
Flatten (128 × 4 × 4 = 2048)
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(1, activation="sigmoid")
    ↓
Output: [ REAL | AI_GENERATED ]   (binary baseline)
```

### Model B: EfficientNet (Transfer Learning)

After the baseline, upgrade to EfficientNet-B0/B3 via transfer learning.

Why EfficientNet:
- Lightweight and efficient
- High accuracy with fewer parameters
- Grad-CAM compatible for explainability
- Easy deployment

```
Image (224x224)
    ↓
EfficientNet-B0 Backbone (pretrained ImageNet)
    ↓
Global Average Pooling
    ↓
Dense (512)
    ↓
Dropout (0.3)
    ↓
Dense (3)
    ↓
Softmax

Output: [ REAL | AI_GENERATED | MANIPULATED ]
```

### Model C: Vision Transformer (ViT)

Captures **global inconsistencies** that CNNs miss.

Why ViT:
- CNN detects local artifacts (strange skin texture, edges)
- ViT detects globally inconsistent structure (facial proportions, lighting direction)

```
Image (224x224)
    ↓
Split into 16x16 patches
    ↓
Patch Embeddings + Position Embeddings
    ↓
Transformer Encoder (12 layers)
    ↓
[CLS] Token
    ↓
Classification Head
    ↓
Softmax

Output: [ REAL | AI_GENERATED | MANIPULATED ]
```

---

## 6. Frequency Domain Detector

This is the **academically interesting** component. AI-generated images contain detectable patterns in the frequency domain.

### Pipeline

```
Image
    ↓
Convert to Grayscale
    ↓
Fast Fourier Transform (FFT)
    ↓
Magnitude Spectrum
    ↓
Extract Frequency Features
    ↓
Classifier (XGBoost / Small Neural Net)
```

### Features Extracted

| Feature | Description |
|---------|-------------|
| Spectral Entropy | Randomness of frequency distribution |
| High-Frequency Energy | Proportion of high-freq components |
| Low-Frequency Energy | Proportion of low-freq components |
| Mean Magnitude | Average FFT magnitude |
| Variance | Spread of frequency magnitudes |
| Radial Distribution | Frequency energy at different radii |
| FFT Artifacts | Periodic spikes from generative upsampling |

### Why This Matters

GANs and diffusion models leave **spectral fingerprints** — periodic patterns in the frequency domain that real photographs do not contain. This signal is often invisible in pixel space but obvious in FFT space.

---

## 7. Metadata Forensics

Most detection systems ignore file metadata entirely. TruthLens extracts and analyzes:

```
EXIF Metadata
│
├── Camera Model ──────────── Is a real camera identified?
├── Software Used ─────────── Photoshop? Stable Diffusion?
├── Creation Date ─────────── Consistent timestamps?
├── GPS Data ──────────────── Geographic consistency?
├── Color Profile ─────────── Standard camera profiles?
├── Compression Info ──────── JPEG quality level?
└── EXIF Presence ─────────── Missing EXIF is suspicious but not definitive
```

### Example Output

```json
{
  "camera_detected": false,
  "editing_software": "Adobe Photoshop 24.0",
  "exif_present": false,
  "compression_ratio": 0.78,
  "suspicious_flags": ["missing_camera_info", "editing_software_detected"]
}
```

**Important:** Missing EXIF does **not** mean AI-generated. Metadata is one signal among many.

---

## 8. Ensemble Decision Engine

Each detector produces a probability distribution. The ensemble combines them.

### Individual Scores

| Model | Real | AI Generated | Manipulated |
|-------|------|-------------|-------------|
| CNN | 0.07 | 0.91 | 0.02 |
| ViT | 0.10 | 0.85 | 0.05 |
| Frequency | 0.15 | 0.76 | 0.09 |
| Metadata | 0.55 | 0.40 | 0.05 |

### Weighted Ensemble

```
Final Score = 0.40 × CNN + 0.30 × ViT + 0.20 × Frequency + 0.10 × Metadata
```

### Verdict Scale

| Score Range | Verdict |
|-------------|---------|
| 0.00 — 0.40 | Likely Real |
| 0.40 — 0.60 | Uncertain |
| 0.60 — 0.80 | Likely AI |
| 0.80 — 1.00 | Highly Likely AI |

**Future improvement:** Replace manual weights with a trained meta-classifier.

---

## 9. Explainable AI (XAI) Layer

### Grad-CAM

Gradient-weighted Class Activation Mapping highlights which regions of the image most influenced the prediction.

```
Input Image
    ↓
CNN Forward Pass
    ↓
Last Convolution Layer Activations
    ↓
Gradient of output w.r.t. activations
    ↓
Weighted sum of activations
    ↓
Heatmap (upsampled to input size)
    ↓
Overlay on Original Image
```

### Output

```
Original Image          Grad-CAM Heatmap

 [Photo of face]    →   [Face with heatmap showing
                         suspicious regions highlighted
                         in red/yellow]
```

Users can **see** which regions the model found suspicious — edges, textures, lighting inconsistencies, or background artifacts.

---

## 10. Robustness Testing Module

AI-generated images shared on social media undergo compression, resizing, and other transformations. A robust detector should maintain consistent predictions.

### Transformations Applied

```
Original Image
    │
    ├── JPEG Compression (Q=50)
    ├── Resize 50%
    ├── Gaussian Noise (σ=0.05)
    └── Random Crop (80%)
```

### Robustness Report

| Transformation | AI Probability | Change |
|----------------|---------------|--------|
| Original | 92% | — |
| JPEG Compression | 89% | -3% |
| Resize 50% | 87% | -5% |
| Gaussian Noise | 83% | -9% |
| Random Crop | 85% | -7% |

### Robustness Score

```
Robustness Score = 1 - std(predictions across transformations)
```

A high robustness score (>0.85) means the model is confident and consistent. A low score suggests the detection is fragile.

---

## 11. Dataset Strategy

### V1.0: CIFAKE (Baseline)

| Property | Value |
|----------|-------|
| Total Images | 120,000 |
| Real Images | 60,000 (CIFAR-10) |
| AI Images | 60,000 (Stable Diffusion v1.4) |
| Resolution | 32x32 RGB |
| Split | 100K train / 20K test |
| Task | Binary: Real vs AI |
| Source | Kaggle / HuggingFace |

### Future Datasets

| Dataset | Scale | Generators | Use Case |
|---------|-------|-----------|----------|
| AIGC Detection Benchmark | Large | 18 models | Cross-generator testing |
| OpenSDI | 300K | SD1.5-3, Flux | Open-world detection |
| TrueFake | 600K | FLUX, SD, StyleGAN | Social media compression |
| DeepShield | 100K | StyleGAN, SD, Flux | Robustness evaluation |

### Critical Research Concept: Cross-Generator Generalization

**Bad experiment:**
```
Train: Stable Diffusion → Test: Stable Diffusion
```
Model learns generator-specific artifacts, not general detection.

**Good experiment:**
```
Train: StyleGAN + Stable Diffusion → Test: Midjourney + DALL-E
```
Tests whether the model can detect an **unseen** generator.

This is called **cross-generator generalization** — and it is the key research contribution.

---

## 12. Technology Stack

### AIML Layer

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Transfer Learning | Keras Applications (EfficientNet), HuggingFace Transformers (TF) |
| Image Processing | OpenCV, Pillow |
| Frequency Analysis | NumPy (FFT) |
| Classical ML | scikit-learn, XGBoost |
| Explainability | Grad-CAM (tf-keras-vis) |
| Experiment Tracking | MLflow (future) |

### Backend (Future)

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Task Queue | Celery + Redis |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Validation | Pydantic |

### Frontend (Future)

| Component | Technology |
|-----------|-----------|
| Framework | Next.js + TypeScript |
| Styling | Tailwind CSS |
| UI Components | Shadcn UI |
| Charts | Recharts |
| State Management | React Query |

### Infrastructure (Future)

| Component | Technology |
|-----------|-----------|
| Containerization | Docker + Docker Compose |
| Reverse Proxy | Nginx |
| Cloud | AWS / Render / HuggingFace Spaces |
| CI/CD | GitHub Actions |

---

## 13. Database Design (Future)

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analysis Results
CREATE TABLE analyses (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    image_path TEXT,
    image_hash VARCHAR(64),

    cnn_score FLOAT,
    vit_score FLOAT,
    frequency_score FLOAT,
    metadata_score FLOAT,

    final_score FLOAT,
    final_prediction VARCHAR(50),

    created_at TIMESTAMP DEFAULT NOW()
);

-- Explainability
CREATE TABLE explanations (
    id UUID PRIMARY KEY,
    analysis_id UUID REFERENCES analyses(id),
    heatmap_path TEXT,
    model_used VARCHAR(50),
    important_regions JSONB
);

-- Model Metrics
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY,
    model_name VARCHAR(100),
    version VARCHAR(20),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 14. API Design (Future)

### Upload & Analyze

```
POST /api/v1/analyze
Content-Type: multipart/form-data

image: <file>

Response:
{
    "analysis_id": "abc123",
    "status": "processing"
}
```

### Get Result

```
GET /api/v1/analysis/{id}

Response:
{
    "analysis_id": "abc123",
    "prediction": "AI_GENERATED",
    "confidence": 0.91,
    "model_scores": {
        "cnn": 0.93,
        "vit": 0.88,
        "frequency": 0.81,
        "metadata": 0.42
    },
    "robustness_score": 0.87,
    "explanation": {
        "heatmap_url": "/heatmaps/abc.png"
    },
    "metadata_analysis": {
        "camera_detected": false,
        "editing_software": null,
        "exif_present": false
    },
    "robustness": {
        "jpeg_compression": 0.89,
        "resize_50": 0.87,
        "gaussian_noise": 0.83,
        "random_crop": 0.85
    }
}
```

---

## 15. Frontend Pages (Future)

```
/
├── Landing Page
│
├── /analyze
│   ├── Image Upload (drag & drop)
│   ├── Analysis Progress (real-time status)
│   └── Result Dashboard
│       ├── Prediction + Confidence
│       ├── Model Score Breakdown (bar chart)
│       ├── Original vs Grad-CAM (side by side)
│       └── Robustness Report
│
├── /dashboard
│   ├── Analysis History (table)
│   ├── Detection Statistics (pie chart)
│   └── Confidence Distribution (histogram)
│
├── /models
│   ├── Model Comparison Table
│   ├── Accuracy Metrics
│   └── Per-Generator Performance
│
└── /about
    ├── Methodology
    └── Research Background
```

---

## 16. Evaluation Metrics

### Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted AI, how many are actually AI |
| Recall | Of actual AI, how many were detected |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve (threshold-independent) |
| Confusion Matrix | TP, TN, FP, FN breakdown |

### System Metrics

| Metric | Description |
|--------|-------------|
| Robustness Score | Prediction stability across transformations |
| Calibration | Does 90% confidence mean ~90% correctness? |
| Generalization | Accuracy on unseen generators |
| Inference Latency | Time per prediction (target: <500ms) |

---

## 17. Development Roadmap

### Phase 0 — Foundation (Current)
- [x] Project description
- [x] Dataset setup (CIFAKE)
- [x] Custom CNN baseline
- [x] Training pipeline
- [x] Evaluation + metrics
- [x] Data exploration notebook

### Phase 1 — Improved Baseline
- [x] Data augmentation pipeline
- [x] Training with augmentation
- [ ] Error analysis (where does the model fail?)
- [ ] Hyperparameter tuning

> **Augmentation experiment result:** horizontal-flip augmentation (`notebooks/Improve_Baseline_2.ipynb`) did **not** improve the baseline — test accuracy 0.9507 / ROC-AUC 0.9896 vs 0.9621 / 0.9933 for the no-augment baseline. Aggressive geometric augmentation (vertical flip / rotation 0.1 / zoom 0.1, earlier attempt) degrades badly (0.888 / 0.959). At 32×32 the AI-vs-real cues are subtle, so augmentation adds variance without improving generalization here. Error analysis and hyperparameter tuning are next before transfer learning.

### Phase 2 — Transfer Learning
- [ ] EfficientNet-B0 fine-tuning
- [ ] ViT fine-tuning
- [ ] Model comparison experiments
- [ ] Cross-generator evaluation

### Phase 3 — Explainability
- [ ] Grad-CAM implementation
- [ ] Heatmap visualization
- [ ] Multi-model explanation comparison

### Phase 4 — Frequency & Metadata
- [ ] FFT feature extraction
- [ ] Frequency domain classifier
- [ ] EXIF metadata analysis
- [ ] Ensemble engine

### Phase 5 — Robustness
- [ ] Transformation pipeline
- [ ] Robustness scoring
- [ ] Adversarial testing

### Phase 6 — Production
- [ ] FastAPI backend
- [ ] Next.js frontend
- [ ] Docker deployment
- [ ] Authentication
- [ ] Analysis history
- [ ] CI/CD pipeline

---

## 18. What Makes This Project Unique

Most student projects follow this pattern:

> Upload image → CNN → Real/Fake → Done

TruthLens AI goes beyond:

| Feature | Typical Project | TruthLens AI |
|---------|----------------|--------------|
| Models | Single CNN | Ensemble (CNN + ViT + FFT + Metadata) |
| Explainability | None | Grad-CAM heatmaps |
| Robustness | Not tested | Systematic transformation testing |
| Generalization | Same generator | Cross-generator evaluation |
| Forensics | Ignored | EXIF + compression analysis |
| Architecture | Notebook only | Production fullstack system |

**Project Title:**

> **TruthLens AI: An Explainable Ensemble Framework for AI-Generated and Manipulated Image Detection**

---

## 19. Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space (for datasets)

### Installation

```bash
git clone https://github.com/yourusername/truthlens-ai.git
cd truthlens-ai
python -m venv .venv
pip install -r requirements.txt
```

### Run Training

Open `notebooks/Baseline_Model_1.ipynb` in Jupyter and run all cells with the `.venv` kernel. It loads CIFAKE (`../data/train`, `../data/test`), trains the baseline CNN, and evaluates it. `notebooks/Improve_Baseline_2.ipynb` is the augmented variant (horizontal flip, `Rescaling`) used for the Phase 1 improvement.

Baseline results (CIFAKE hold-out test set, 32×32, batch 32, `notebooks/Baseline_Model_1.ipynb`):
- Test accuracy: 0.9621
- ROC-AUC: 0.9933
- Peak validation accuracy during training: ~0.965

Augmented baseline (`notebooks/Improve_Baseline_2.ipynb`, horizontal flip):
- Test accuracy: 0.9507
- ROC-AUC: 0.9896
- Peak validation accuracy during training: ~0.958

*(CLI training/evaluation scripts will be added in a later phase.)*

---

## 20. References

1. Bird, J.J. & Lotfi, A. (2023). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. arXiv:2303.14126.
2. Rossler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV 2019.
3. Li, Y. et al. (2020). Celeb-DF: A Large-scale Challenging Dataset for Deepfake Forensics. CVPR 2020.
4. Wang, Y. et al. (2025). OpenSDI: Spotting Diffusion-Generated Images in the Open World. CVPR 2025.
5. Dell'Anna, S. et al. (2025). TrueFake: A Real World Case Dataset. IJCNN 2025.
