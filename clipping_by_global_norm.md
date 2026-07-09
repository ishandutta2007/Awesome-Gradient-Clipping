# B. Clipping by Global Norm ($L_2$ Cap)

Evaluating the aggregate Euclidean norm of the entire system.

## Architecture Diagram

```mermaid
flowchart LR
    A[Sum of squared gradients] --> B[Square root (L2 Norm)] --> C[Scale entire matrix if > threshold]
```

[Back to README](README.md)
