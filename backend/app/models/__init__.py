from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.cnn_tw3 import CNNTW3
from app.models.ensemble_cnn import EnsembleCNN
from app.models.mask_rcnn import MaskRCNNRegression
from app.models.multimodal_cnn import MultimodalCNN
from app.models.multiscale_cnn_rf import (
    MultiscaleCNNFeatureExtractor,
    MultiscaleCNNRFWrapper,
)
from app.models.regression_multimodal import RegressionMultimodal

MODEL_REGISTRY = {
    "cnn": {
        "class": CNNBaseline,
        "display_name": "CNN Baseline",
        "description": "ResNet-18 backbone with linear regression head",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "cnn_dnn": {
        "class": CNNDNN,
        "display_name": "CNN + DNN",
        "description": "ResNet-18 with multi-layer dense regression head",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "multimodal_cnn": {
        "class": MultimodalCNN,
        "display_name": "Multimodal CNN",
        "description": "Image CNN fused with gender metadata",
        "requires_gender": True,
        "supports_gradcam": True,
    },
    "regression_multimodal": {
        "class": RegressionMultimodal,
        "display_name": "Regression-Based Multimodal DL",
        "description": "CNN image features fused with gender metadata via regression head",
        "requires_gender": True,
        "supports_gradcam": True,
    },
    "mask_rcnn": {
        "class": MaskRCNNRegression,
        "display_name": "Mask R-CNN Regression Proxy",
        "description": "Proxy Mask R-CNN-style region guided regressor for bone age",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "ensemble_cnn": {
        "class": EnsembleCNN,
        "display_name": "Ensemble CNN",
        "description": "Multiple CNN branches fused into a shared regression head",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "cnn_tw3": {
        "class": CNNTW3,
        "display_name": "CNN + TW3 Proxy",
        "description": "CNN features fused with TW3-inspired skeletal proxy features",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "cnn_rf": {
        "class": CNNFeatureExtractor,
        "display_name": "CNN + Random Forest",
        "description": "CNN feature extractor with sklearn RF regressor",
        "requires_gender": False,
        "supports_gradcam": True,
        "wrapper": CNNWithRFWrapper,
    },
    "multiscale_cnn_rf": {
        "class": MultiscaleCNNFeatureExtractor,
        "display_name": "Multiscale CNN + Random Forest",
        "description": "Multiscale CNN feature extractor with sklearn RF regressor",
        "requires_gender": False,
        "supports_gradcam": True,
        "wrapper": MultiscaleCNNRFWrapper,
    },
}

SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())
