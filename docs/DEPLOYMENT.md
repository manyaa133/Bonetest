# Deployment Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- (Training) NVIDIA GPU with CUDA, or Kaggle/Colab runtime

## Local Development

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env

# Create demo checkpoints for local testing
python scripts/create_demo_checkpoints.py

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173

---

## Training on Kaggle

1. Add RSNA Pediatric Bone Age dataset to your notebook.
2. Upload `backend/ml/` and `backend/app/models/` to Kaggle.
3. Run:

```python
!cd /kaggle/working/backend && python -m ml.train --model-type cnn --data-dir /kaggle/input/rsna-bone-age --epochs 30 --pretrained
!python -m ml.train --model-type cnn_dnn --data-dir /kaggle/input/rsna-bone-age --epochs 30 --pretrained
!python -m ml.train --model-type multimodal_cnn --data-dir /kaggle/input/rsna-bone-age --epochs 30 --pretrained
!python -m ml.train --model-type cnn_rf --data-dir /kaggle/input/rsna-bone-age --epochs 30 --pretrained
!python -m ml.evaluate --data-dir /kaggle/input/rsna-bone-age
!python -m ml.compare_models
```

4. Download `checkpoints/`, `rf_models/`, and `metrics/` artifacts.

## Training on Google Colab

See `notebooks/train_colab.ipynb`. Enable GPU: Runtime → Change runtime type → T4 GPU.

---

## Production Deployment

### Docker (Backend)

```bash
cd backend
python scripts/create_demo_checkpoints.py   # or copy trained checkpoints
docker build -t bone-age-api .
docker run -p 8000:8000 -e DEVICE=cpu bone-age-api
```

### Docker Compose (Full Stack)

```bash
docker compose up --build
```

### Frontend (Static)

```bash
cd frontend
npm run build
# Serve dist/ with Nginx, Vercel, or Netlify
# Set VITE_API_URL to your backend URL at build time
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHECKPOINT_DIR` | Model weights directory | `checkpoints` |
| `METRICS_DIR` | Evaluation JSON/plots | `metrics` |
| `DEVICE` | `cpu` or `cuda` | `cpu` |
| `CORS_ORIGINS` | Allowed frontend origins | `localhost:5173` |
| `VITE_API_URL` | Backend URL (frontend build) | `http://localhost:8000` |

### GPU Inference

Set `DEVICE=cuda` and use a CUDA-enabled PyTorch base image. Mount checkpoints as a volume for updates without rebuild.

---

## Checklist Before Production

- [ ] Replace demo checkpoints with trained weights
- [ ] Update `NORM_MEAN` / `NORM_STD` from `checkpoints/normalization.json`
- [ ] Run full evaluation and verify metrics
- [ ] Configure HTTPS and authentication if exposed publicly
- [ ] Add rate limiting and file size caps
- [ ] Display "research use only" disclaimer (included in frontend footer)
