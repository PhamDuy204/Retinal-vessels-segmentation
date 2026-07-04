from __future__ import annotations

import modal
from modal_statistics import CODE_ROOT, image

app = modal.App("retinal-ournet-param-count")


@app.function(image=image, timeout=600)
def count_params(widths: list[int]) -> list[tuple[int, int]]:
    import sys

    sys.path.insert(0, CODE_ROOT)
    from load_model import load_model_class
    from utils import count_trainable_params

    model_class = load_model_class("our_net")
    results = []
    for width in widths:
        model = model_class(1, 1, width=width)
        results.append((width, count_trainable_params(model)))
    return results


@app.local_entrypoint()
def main() -> None:
    for width, params in count_params.remote([64, 80, 88, 96, 104]):
        print(f"width={width} params={params:,}")
