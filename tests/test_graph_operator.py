import torch

from deepuq.models import GraphNeuralOperator2D


def test_graph_operator_forward_shape() -> None:
    model = GraphNeuralOperator2D(in_channels=2, hidden_dim=16, message_dim=12)
    x = torch.randn(3, 8, 8, 2)
    y = model(x)
    assert y.shape == (3, 8, 8, 2)
    assert torch.isfinite(y).all()


def test_graph_operator_neighbor_graph_has_expected_connectivity() -> None:
    model = GraphNeuralOperator2D(
        in_channels=2,
        hidden_dim=8,
        message_dim=8,
        radius=1,
    )
    graph = model._build_graph(4, 4, device=torch.device("cpu"), dtype=torch.float32)
    assert graph.src.shape == graph.dst.shape
    assert graph.edge_attr.shape[0] == graph.src.numel()
    assert graph.coords.shape == (16, 2)
    assert graph.src.numel() > 0
    center_idx = 1 * 4 + 1
    incoming_center = (graph.dst == center_idx).sum().item()
    assert incoming_center == 8


def test_graph_operator_coordinate_features_break_symmetry() -> None:
    model = GraphNeuralOperator2D(in_channels=2, hidden_dim=12, message_dim=12)
    zeros = torch.zeros(1, 10, 10, 2)
    out = model(zeros)
    assert out.std() > 0.0
