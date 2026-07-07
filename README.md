# Bone Age Predictor

Full-stack AI application for pediatric bone age regression from hand X-ray images. Compares four deep learning architectures with Grad-CAM explainability.

## Architecture

| Component | Stack |
|-----------|-------|
| Frontend | React 18, TypeScript, Tailwind CSS, Axios, Recharts |
| Backend | FastAPI, PyTorch, OpenCV, scikit-learn |
| Training | Kaggle / Google Colab (GPU) |
| Inference | Saved checkpoints on CPU/GPU |

## Models

1. **CNN** — ResNet-18 backbone + linear regression head
2. **CNN + DNN** — ResNet-18 + multi-layer dense head
3. **Multimodal CNN** — Image + gender metadata fusion
4. **CNN + Random Forest** — CNN feature extractor + sklearn RF regressor

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

App: http://localhost:5173

## Training (Kaggle / Colab)

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) and `notebooks/train_colab.ipynb`.

```bash
cd backend
python -m ml.train --model-type cnn --data-dir /path/to/rsna --epochs 30
python -m ml.evaluate --data-dir /path/to/rsna --checkpoints-dir checkpoints
python -m ml.compare_models
```

Copy generated `checkpoints/` and `metrics/` into the backend before deployment.

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## Project Structure

```
bonetest/
├── backend/          # FastAPI inference server + ML pipeline
├── frontend/         # React SPA
├── docs/             # Architecture & deployment docs
└── notebooks/        # Colab/Kaggle training notebook
```

## License

MIT — RSNA dataset subject to Kaggle competition terms.
