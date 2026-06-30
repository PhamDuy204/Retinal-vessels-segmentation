# `our_net` optimization benchmark

Measured on an NVIDIA GeForce RTX 5060 Ti 16 GB with PyTorch 2.7.1+cu128 and cuDNN 9.7.1. The loaded model was `/workspace/Retinal-vessels-segmentation/models/our_net/our_net.py` with 964,679 trainable parameters.

## Production profiler

The first ten optimizer steps used FP32, `abe_dice_loss_optimized`, patch size 64, the unchanged 750 sampled patches per image, batch size 1 at the image-loader level, two persistent workers, and pinned memory. Profiling adds substantial overhead, so use these values to identify proportions rather than normal epoch throughput.

| Phase | Calls | CPU time | CUDA time |
|---|---:|---:|---:|
| Data loading/worker startup | 1 | 7,466.6 ms | 0 ms |
| Model forward | 10 | 5,567.0 ms | 5,929.5 ms |
| Loss forward | 10 | 110.9 ms | 100.4 ms |
| Backward | 10 | 10,697.7 ms | 10,426.3 ms |
| Optimizer step | 10 | 277.3 ms | 216.0 ms |
| Full 20-image evaluation | 1 | 60,331.3 ms | 60,330.1 ms |

Backward and model forward dominate. Optimizer work and loss evaluation are comparatively small after the zero-weight loss scopes are removed.

## Exact-safe benchmark

This benchmark used micro-batch 8, three warmup steps, ten measured steps and one evaluation image. Epoch compute time is extrapolated from the median step and excludes data loading.

| Scenario | Median step | Epoch compute | Evaluation | Peak VRAM | Throughput |
|---|---:|---:|---:|---:|---:|
| Original loss baseline | 137.437 ms | 257.69 s | 2.685 s | 1080.7 MiB | 58.21 patches/s |
| Optimized loss | 135.679 ms | 254.40 s | 2.542 s | 1074.9 MiB | 58.96 patches/s |
| Optimized loss + non-blocking transfer | 135.983 ms | 254.97 s | 2.542 s | 1074.9 MiB | 58.83 patches/s |

Preprocessing of one image was 31.10 ms cold and 0.017 ms from the RAM cache. The original and optimized loss values were bit-identical on the same predictions (absolute difference `0.0`). Accuracy, F1, IoU, recall, specificity, AUC and Dice were identical to strict tolerance on the same evaluation outputs.

## Experimental modes

AMP and compile were measured separately with one warmup and three timed steps, so compare them directionally rather than mixing the numbers into the exact-safe table.

| Mode | Median step | Peak VRAM | Throughput | Observation |
|---|---:|---:|---:|---|
| AMP | 165.935 ms | 902.7 MiB | 48.21 patches/s | Lower memory, slower on this GPU/model |
| `torch.compile` | 120.561 ms | 962.4 MiB | 66.36 patches/s | Approximately 13.5% faster than that run's 139.389 ms baseline |

AMP, TF32 and compile can alter floating-point behavior and remain disabled by default. Re-run `benchmark_training.py` on the target hardware before enabling them for a statistical sweep.
