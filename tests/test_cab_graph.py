import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "our_net"))
from bottle_neck import CAB_1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cab_can_replay_graph_with_new_inputs():
    torch.manual_seed(42)
    model = CAB_1(32).cuda().eval()
    x = torch.randn(2, 32, 8, 8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            model(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            actual = model(x)
        x.copy_(torch.randn_like(x))
        expected = model(x)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
