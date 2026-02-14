# H-JEPA: Learning Hamiltonian Dynamics on Latent Video Embeddings

**Jefrey Bergl** — University of North Carolina at Chapel Hill

H-JEPA enforces Hamiltonian mechanics in the latent space of a pre-trained video encoder ([V-JEPA 2](https://github.com/facebookresearch/vjepa)). Instead of learning an unconstrained next-state predictor, it learns a scalar energy function H(q, p) and evolves states via a symplectic Stormer-Verlet (leapfrog) integrator — guaranteeing bounded energy drift over long rollouts by construction.

## Key idea

1. **Encode** video clips into 1024-d latent vectors using V-JEPA 2 (ViT-L, frozen).
2. **Split** each latent z into canonical coordinates q and momenta p (512-d each).
3. **Learn** a Hamiltonian H(q, p) → R as a small MLP (~0.9M params).
4. **Integrate** dynamics via leapfrog with a learnable timestep dt, preserving the symplectic 2-form exactly.

## Results

On synthetic 2-ball elastic collision videos:

| Model | MSE | Energy Drift | Params |
|---|---|---|---|
| **H-JEPA (full)** | **best** | **best** | 0.9M |
| HNN (no energy loss) | +ablation | worse | 0.9M |
| HNN (Euler integrator) | +ablation | worse | 0.9M |
| Baseline MLP | 1.9x higher | higher | 1.3M |

Both the symplectic integrator and energy conservation loss contribute to performance. See the paper for full numbers and ablation analysis.

## Repository contents

```
hjepanotebook.ipynb          # Full pipeline: data generation, V-JEPA 2 extraction,
                             #   HNN training, ablations, evaluation, and plots
hjepapaper/
  iclr2025_conference.pdf    # Paper
```

## Running the notebook

The notebook is designed for **Google Colab with a GPU runtime** (T4 or better). It will:

1. Generate 50 synthetic collision videos
2. Extract V-JEPA 2 latents (requires ~3GB VRAM)
3. Train the HNN + ablation models (50 epochs each)
4. Produce all figures and the summary table

Install dependencies (handled in the first cell):
```
pip install timm einops av opencv-python-headless torchcodec transformers
```

## Citation

```bibtex
@article{bergl2025hjepa,
  title={H-JEPA: Learning Hamiltonian Dynamics on Latent Video Embeddings},
  author={Bergl, Jefrey},
  year={2025}
}
```

## License

MIT
