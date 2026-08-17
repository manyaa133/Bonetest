import torch

from app.models import MODEL_REGISTRY
from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.multimodal_cnn import MultimodalCNN
from app.models.new_models import (
    RegressionMultimodal,
    EnsembleCNN,
    MultiscaleCNNRFWrapper,
    CNNTW3,
    MaskRCNNRegression,
)


def test_model_registry_contains_new_models():
    assert "regression_multimodal" in MODEL_REGISTRY
    assert "ensemble_cnn" in MODEL_REGISTRY
    assert "multiscale_cnn_rf" in MODEL_REGISTRY
    assert "cnn_tw3" in MODEL_REGISTRY
    assert "mask_rcnn" in MODEL_REGISTRY


def test_new_models_forward_pass():
    batch = torch.randn(2, 1, 64, 64)
    gender = torch.tensor([[1.0], [0.0]])

    regression_model = RegressionMultimodal(pretrained=False)
    ensemble_model = EnsembleCNN(pretrained=False, num_branches=2)
    multiscale_model = MultiscaleCNNRFWrapper(pretrained=False)
    tw3_model = CNNTW3(pretrained=False)
    mask_model = MaskRCNNRegression(pretrained=False)

    assert regression_model(batch, gender).shape == torch.Size([2])
    assert ensemble_model(batch).shape == torch.Size([2])
    assert multiscale_model.cnn(batch).shape == torch.Size([2, 512])
    assert tw3_model(batch).shape == torch.Size([2])
    assert mask_model(batch).shape == torch.Size([2])


def test_existing_models_still_work():
    assert isinstance(CNNBaseline(pretrained=False), torch.nn.Module)
    assert isinstance(CNNDNN(pretrained=False), torch.nn.Module)
    assert isinstance(MultimodalCNN(pretrained=False), torch.nn.Module)
