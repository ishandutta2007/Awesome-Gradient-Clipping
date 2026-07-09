# C. Adaptive Gradient Clipping (AGC)

Scales gradients relative to active weight tensors on a layer-by-layer basis.

## Architecture Diagram

```mermaid
flowchart LR
    A[Layer L Gradients] --> B[Layer L Weights] --> C[Scale factor lambda * ||W|| / ||G||]
```

[Back to README](README.md)
