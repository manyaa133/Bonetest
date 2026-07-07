# Bone Age Predictor — Implementation Plan

## 1. Overview

Bone Age Predictor is a full-stack regression system that estimates pediatric skeletal maturity (bone age in months) from hand/wrist X-ray images. The system compares four model architectures, exposes REST APIs for inference and explainability, and provides a React dashboard for clinical-style review.

**Dataset:** [RSNA Pediatric Bone Age Challenge](https://www.kaggle.com/competitions/rsna-bone-age) (Kaggle). CSV columns: `id`, `boneage`, `male`. Images: `{id}.png`.

**Problem formulation:** Supervised regression — predict continuous bone age (months) from grayscale X-ray.

**Deployment split:**
- **Training** (Kaggle/Colab GPU): full pipeline, checkpoint export
- **Production backend** (CPU/GPU): load checkpoints, inference + Grad-CAM only

---

## 2. System Architecture

```
┌─────────────────┐     REST/JSON      ┌──────────────────────────────┐
│  React Frontend │ ◄────────────────► │  FastAPI Backend             │
│  (TypeScript)   │   multipart/form   │  PyTorch inference + Grad-CAM│
└─────────────────┘                    └──────────────────────────────┘
                                                    │
                                         ┌──────────┴──────────┐
                                         │  checkpoints/*.pt   │
                                         │  rf_models/*.joblib │
                                         │  metrics/*.json     │
                                         └─────────────────────┘
```

**Clean architecture layers (backend):**
| Layer | Responsibility |
|-------|----------------|
| `api/routes` | HTTP, validation, response mapping |
| `schemas` | Pydantic request/response contracts |
| `services` | Business logic: inference, Grad-CAM, metrics |
| `models` | PyTorch module definitions |
| `ml` | Offline training, preprocessing, evaluation |

---

## 3. Dataset Preprocessing Pipeline

### 3.1 Loading
1. Read `train.csv` / validation split CSV.
2. Resolve image paths `{data_dir}/{id}.png`.
3. Optional multimodal fields: `male` (0/1), patient id.

### 3.2 Image preprocessing
| Step | Details |
|------|---------|
| Read | OpenCV `imread` grayscale |
| Resize | 512×512 (maintain aspect via letterbox pad) |
| Normalize | Per-image z-score OR dataset mean/std (saved in `normalization.json`) |
| Tensor | `(1, H, W)` float32 |

### 3.3 Augmentation (training only)
- Random horizontal flip (50%)
- Random rotation ±10°
- Random brightness/contrast (±15%)
- Gaussian noise (σ=0.01)
- Random crop 90–100% scale

Implemented in `ml/augmentation.py` using Albumentations-style OpenCV transforms.

### 3.4 Train/val split
- 85/15 stratified by age bins (0–60, 60–120, 120–216 months)
- Seed=42 for reproducibility

---

## 4. Model Architectures

Shared CNN backbone: **ResNet-18** (ImageNet weights optional at train time; inference uses saved weights).

Input: `(B, 1, 512, 512)` — first conv adapted to 1 channel.

### 4.1 Model A — CNN Baseline
```
Input → ResNet18 (modified) → GAP → Linear(512→1) → bone age (months)
```
- Direct end-to-end regression
- Grad-CAM target layer: `layer4[-1].conv2`

### 4.2 Model B — CNN + DNN
```
Input → ResNet18 → GAP → Flatten(512)
     → FC(512→256) → BN → ReLU → Dropout(0.3)
     → FC(256→128) → BN → ReLU → Dropout(0.2)
     → FC(128→1)
```
- Deeper non-linear head for refined regression
- Grad-CAM: same backbone hook as Model A

### 4.3 Model C — Multimodal CNN
```
Image branch: ResNet18 → GAP → 512-d embedding
Meta branch:  [male, optional_height] → FC(2→32) → ReLU → 32-d
Fusion:       concat(512+32) → FC(544→256) → ... → 1
```
- Gender strongly correlates with bone age curves (Tanner-Whitehouse)
- At inference: gender required in API form field
- Grad-CAM: image branch only (metadata path has no spatial map)

### 4.4 Model D — CNN + Random Forest
**Two-stage:**
1. **Feature extraction:** Train/fine-tune CNN with regression head (same as A), then replace head with identity on 512-d GAP features.
2. **RF training:** Extract 512-d features for all train/val samples; fit `RandomForestRegressor(n_estimators=200, max_depth=20)`.
3. **Inference:** CNN forward to features → `rf.predict(features)`.
- Grad-CAM: CNN backbone (same hook); RF is non-differentiable but explanation uses CNN spatial gradients

**Checkpoint artifacts:**
- `cnn.pt`, `cnn_dnn.pt`, `multimodal_cnn.pt` — full PyTorch state dict
- `cnn_rf.pt` — CNN feature extractor + `cnn_rf.joblib` — sklearn RF

---

## 5. Training Pipeline

Location: `backend/ml/train.py` + `notebooks/train_colab.ipynb`

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW, lr=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=epochs |
| Loss | Smooth L1 (Huber) |
| Batch size | 16 (GPU), 8 (Colab free tier) |
| Epochs | 30 (early stop patience=5 on val MAE) |
| Mixed precision | AMP on CUDA |

**Loop:**
1. Build dataloaders with augmentation (train) / no aug (val)
2. Instantiate model by `--model-type`
3. Train epoch → validate → save best checkpoint by val MAE
4. Log metrics to CSV/JSON
5. Export `checkpoints/{model}_best.pt`

**Colab/Kaggle entry:**
```bash
python -m ml.train --model-type cnn --data-dir /kaggle/input/... --epochs 30
```

---

## 6. Validation & Evaluation Pipeline

`ml/evaluate.py`:
1. Load each checkpoint on held-out test set
2. Compute per-sample predictions
3. Aggregate metrics (see §7)
4. Save `metrics/evaluation_results.json` and `metrics/predictions_{model}.csv`
5. Generate plots → `metrics/plots/` (PNG, also served via API as base64 or static files)

---

## 7. Evaluation Metrics

For each model on validation/test set:

| Metric | Formula |
|--------|---------|
| MAE | `mean(|y - ŷ|)` |
| MSE | `mean((y - ŷ)²)` |
| RMSE | `sqrt(MSE)` |

**Comparison outputs:**
- JSON table for API consumption
- Bar charts (MAE, MSE, RMSE side-by-side)
- Scatter plot: predicted vs actual per model
- Summary CSV for publication

---

## 8. Inference Pipeline (Production)

`services/inference_service.py`:
1. Receive uploaded image bytes
2. Preprocess (same as val: resize, normalize using saved stats)
3. Load model from registry (lazy singleton per model type)
4. `torch.no_grad()` forward pass
5. Return bone age (months), confidence (derived from val RMSE inverse or prediction interval)
6. Optional gender for multimodal

**Confidence score:** `confidence = max(0, 1 - |pred - cohort_mean| / (3 * cohort_std))` clipped, or fixed per-model calibration from validation residuals.

---

## 9. Grad-CAM (Explainable AI)

`services/gradcam_service.py`:
1. Forward pass with `requires_grad=True` on input
2. Hook last conv layer activations + gradients
3. Weights = global average pool of gradients
4. CAM = ReLU(Σ w_k · A_k), upsample to input size
5. Overlay heatmap on original X-ray (JET colormap, α=0.45)
6. Return base64 PNG for frontend

Applied to: CNN, CNN+DNN, Multimodal (image branch), CNN+RF (feature CNN).

---

## 10. REST API Design

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Liveness |
| GET | `/api/models` | List available models + loaded status |
| POST | `/api/predict` | multipart: `file`, `model_type`, optional `gender` |
| POST | `/api/gradcam` | Same + returns heatmap image |
| GET | `/api/metrics` | Full evaluation JSON |
| GET | `/api/metrics/comparison` | Summary table |
| GET | `/api/metrics/plots/{name}` | PNG plot |

---

## 11. Frontend Structure

```
frontend/src/
├── api/client.ts          # Axios instance
├── components/
│   ├── ImageUpload.tsx    # Drag-and-drop
│   ├── PredictionResult.tsx
│   ├── GradCamViewer.tsx
│   ├── ModelSelector.tsx
│   ├── MetricsChart.tsx
│   └── ComparisonDashboard.tsx
├── pages/
│   ├── PredictPage.tsx
│   └── DashboardPage.tsx
├── hooks/usePrediction.ts
└── App.tsx
```

**UX flow:**
1. User drops X-ray → preview
2. Select model + optional gender
3. Predict → show months + confidence
4. Toggle Grad-CAM overlay
5. Dashboard tab: load metrics, bar charts (Recharts), comparison table

---

## 12. Integration & Deployment

| Environment | Frontend | Backend |
|-------------|----------|---------|
| Dev | `npm run dev` :5173 | `uvicorn app.main:app` :8000 |
| Prod | `npm run build` → Nginx | Docker + Gunicorn/Uvicorn |

CORS configured for frontend origin. `.env` for `VITE_API_URL`, `CHECKPOINT_DIR`.

**Docker:** Multi-stage — backend image with PyTorch CPU; frontend nginx serving static + proxy `/api`.

---

## 13. File Deliverables Checklist

- [x] Implementation plan (this document)
- [ ] Backend FastAPI application
- [ ] Four PyTorch/sklearn model definitions
- [ ] ML training/evaluation scripts
- [ ] Sample metrics + placeholder checkpoints
- [ ] React frontend with all UI features
- [ ] README, API docs, deployment guide
- [ ] Colab training notebook

---

## 14. Implementation Order

1. Backend model definitions + preprocessing
2. Inference + Grad-CAM services
3. API routes + sample metrics
4. Frontend scaffold + API integration
5. Training scripts + notebook
6. Documentation + Docker
