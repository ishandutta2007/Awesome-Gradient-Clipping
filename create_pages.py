import os
import re

os.chdir(r"C:\Users\ishan\Documents\Projects\Awesome-Gradient-Clipping")

files_data = [
    ('element_wise_value_truncation.md', 'The Element-Wise Value Truncation Era', "A detailed look at the early days of gradient clipping using coordinate truncation.", 'flowchart TD\n    A[Gradient Element] --> B{> Threshold c?}\n    B -- Yes --> C[Set to c]\n    B -- No --> D{< -c?}\n    D -- Yes --> E[Set to -c]\n    D -- No --> F[Keep Original]'),
    ('global_vector_norm_capping.md', 'The Global Vector Norm Capping Era', "Exploring the transition to global vector norm constraints introduced by Pascanu et al.", 'flowchart TD\n    A[Compute Total L2 Norm] --> B{Norm > Cap v?}\n    B -- Yes --> C[Scale Matrix by v/Norm]\n    B -- No --> D[Keep Gradients Unchanged]'),
    ('adaptive_layer_wise_normalization.md', 'The Adaptive Layer-Wise Normalization Era', "Adaptive Gradient Clipping (AGC) allows scaling gradients relative to active weight tensors.", 'flowchart LR\n    A[Layer Weights] --> B[Compute Weight Norm]\n    C[Layer Gradients] --> D[Compute Gradient Norm]\n    B & D --> E[Ratio: WeightNorm/GradNorm]\n    E --> F[Scale Layer Gradients]'),
    ('per_sample_differential_privacy.md', 'The Per-Sample Differential Privacy Era', "Incorporating differential privacy by moving clipping from global batch down to individual samples.", 'flowchart TD\n    A[Sample 1 Gradient] --> B[Clip L2 Norm]\n    C[Sample N Gradient] --> D[Clip L2 Norm]\n    B & D --> E[Aggregate]\n    E --> F[Add Calibrated Gaussian Noise]'),
    ('clipping_by_value.md', 'A. Clipping by Value (Element-Wise Truncation)', "Mechanisms and tradeoffs of clamping values to a strict minimum/maximum interval.", 'flowchart LR\n    A[Raw Gradient Matrix] --> B[Coordinate-wise Scan] --> C[Clamp between c_min and c_max]'),
    ('clipping_by_global_norm.md', 'B. Clipping by Global Norm ($L_2$ Cap)', "Evaluating the aggregate Euclidean norm of the entire system.", 'flowchart LR\n    A[Sum of squared gradients] --> B[Square root (L2 Norm)] --> C[Scale entire matrix if > threshold]'),
    ('adaptive_gradient_clipping.md', 'C. Adaptive Gradient Clipping (AGC)', "Scales gradients relative to active weight tensors on a layer-by-layer basis.", 'flowchart LR\n    A[Layer L Gradients] --> B[Layer L Weights] --> C[Scale factor lambda * ||W|| / ||G||]'),
    ('per_sample_privacy_clipping.md', 'D. Per-Sample Privacy Clipping', "Isolating each data row's forward-backward pass to execute individual gradient clipping.", 'flowchart TD\n    A[Forward/Backward Pass 1] --> B[Clip 1]\n    C[Forward/Backward Pass 2] --> D[Clip 2]\n    B & D --> E[Batch Reduction]'),
    ('vectorized_opacus_kernels.md', 'Vectorized Opacus Kernels', "Overcoming memory bottlenecks with specialized JIT-compilers like PyTorch Opacus.", 'flowchart LR\n    A[Traditional Loop] --> B[Memory Bottleneck]\n    C[Vectorized Opacus] --> D[Fast GPU SRAM Execution]\n    B & D -.-> E[Efficiency Comparison]'),
    ('global_norm_threshold_caching.md', 'Global Norm Threshold ($\\text{grad}\\_\\text{norm}$) Caching', "Memory bus load balancing and single scalar tensor caching.", 'flowchart TD\n    A[Data Nodes] --> B[Accumulate Partial Norms]\n    B --> C[All-Reduce Scalar]\n    C --> D[Cache Global Norm Threshold]'),
    ('distributed_all_reduce_stale_wait.md', 'The Distributed All-Reduce Communication Stale Wait', "Understanding the sync barrier stalls in FSDP and distributed configurations.", 'flowchart LR\n    A[GPU 1] --> C[Sync Barrier All-Reduce]\n    B[GPU 2] --> C\n    C --> D[Global Norm Computed]\n    D --> E[Resume Operation]'),
    ('low_precision_gradient_underflow.md', 'The Low-Precision Gradient Underflow Hazard', "Handling long-tail parameter update underflow in FP16/BF16 routines.", 'flowchart LR\n    A[Low Precision Gradients] --> B[Clip Factors Applied]\n    B --> C{Value < numerical limit?}\n    C -- Yes --> D[Underflow to Zero]\n    C -- No --> E[Update Preserved]'),
    ('pre_training_trillion_token.md', 'Pre-Training Trillion-Token Foundational LLM Backbones (Llama / DeepSeek)', "Baseline safety guards protecting distributed clusters from optimization divergence.", 'flowchart TD\n    A[Uncurated Token Batch] --> B{Anomalous Spike?}\n    B -- Yes --> C[Global Norm Clip intercepts]\n    C --> D[Loss Stability Maintained]'),
    ('privacy_certified_medical.md', 'Privacy-Certified Medical & Financial Foundational Fine-Tuning', "Ensuring internal patterns are safely abstracted within certified differential privacy bounds.", 'flowchart LR\n    A[Sensitive Enterprise Data] --> B[Per-sample Clip & Noise]\n    B --> C[Provably Private Model]'),
    ('high_frequency_multi_task.md', 'High-Frequency Multi-Task Autonomous Perception Stacks', "Coordinating real-time navigation pipelines for autonomous vehicles.", 'flowchart TD\n    A[Object Tracking] --> D[AGC Stabilization]\n    B[Lane Segmentation] --> D\n    C[Depth Calculations] --> D\n    D --> E[Multi-task Parameter Optimization]')
]

readme_content = open('README.md', 'r', encoding='utf-8').read()

for filename, title, desc, diagram in files_data:
    content = f"# {title}\n\n{desc}\n\n## Architecture Diagram\n\n```mermaid\n{diagram}\n```\n\n[Back to README](README.md)\n"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Replace in README
    search_str = f"**{title}**"
    replace_str = f"[**{title}**]({filename})"
    if search_str in readme_content:
        readme_content = readme_content.replace(search_str, replace_str)
    else:
        # If the exact title contains dollar signs, we might need a regex or exact string fallback
        readme_content = readme_content.replace(search_str.replace('\\', '\\\\'), replace_str)

# Fallbacks for any that might have been missed due to slight text differences
fallback_replacements = {
    '**B. Clipping by Global Norm ($L_2$ Cap)**': f'[**B. Clipping by Global Norm ($L_2$ Cap)**](clipping_by_global_norm.md)',
    '**Global Norm Threshold ($\\text{grad}\\_\\text{norm}$) Caching**': '[**Global Norm Threshold ($\\text{grad}\\_\\text{norm}$) Caching**](global_norm_threshold_caching.md)'
}
for k, v in fallback_replacements.items():
    readme_content = readme_content.replace(k, v)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print('Detailed pages created and linked in README')
