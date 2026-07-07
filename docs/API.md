# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `/docs` (Swagger UI) · `/redoc` (ReDoc)

## Endpoints

### GET `/api/health`

Liveness check.

**Response**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": ["cnn"]
}
```

---

### GET `/api/models`

List available models and checkpoint status.

**Response**
```json
{
  "models": [
    {
      "id": "cnn",
      "display_name": "CNN Baseline",
      "description": "ResNet-18 backbone with linear regression head",
      "requires_gender": false,
      "supports_gradcam": true,
      "checkpoint_available": true
    }
  ],
  "default_model": "cnn"
}
```

---

### POST `/api/predict`

Predict bone age from an uploaded X-ray.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | PNG/JPEG hand X-ray |
| `model_type` | string | no | `cnn`, `cnn_dnn`, `multimodal_cnn`, `cnn_rf` (default: `cnn`) |
| `gender` | string | conditional | `male` or `female` — required for `multimodal_cnn` |

**Response**
```json
{
  "model_type": "cnn",
  "bone_age_months": 127.4,
  "confidence": 0.87,
  "gender_used": null,
  "processing_time_ms": 142.5
}
```

---

### POST `/api/gradcam`

Run prediction and generate Grad-CAM heatmap.

Same form fields as `/api/predict`.

**Response**
```json
{
  "model_type": "cnn",
  "bone_age_months": 127.4,
  "confidence": 0.87,
  "heatmap_base64": "<base64 PNG>",
  "overlay_base64": "<base64 PNG>",
  "processing_time_ms": 312.0
}
```

---

### GET `/api/metrics`

Full evaluation results including comparison table and plot paths.

---

### GET `/api/metrics/comparison`

Summary comparison table only.

---

### GET `/api/metrics/plots/{plot_name}`

Returns PNG image. Available after running `ml.evaluate`:
- `metrics_comparison`
- `scatter_comparison`

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Invalid input (bad image, missing gender) |
| 404 | Plot not found |
| 503 | Model checkpoint not available |
| 500 | Internal inference error |
