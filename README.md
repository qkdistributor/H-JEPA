# H-JEPA: Learning Hamiltonian Dynamics on Latent Video Embeddings

H-JEPA takes a pre-trained video encoder ([V-JEPA 2](https://github.com/facebookresearch/vjepa)) and enforces Hamiltonian mechanics in its latent space. Rather than learning some unconstrained next-state predictor, we learn a scalar energy function H(q, p) and evolve states with a symplectic leapfrog integrator. This gives us bounded energy drift over long rollouts, by construction.

## Key idea

1. **Encode** video clips into 1024-d latent vectors using V-JEPA 2 (ViT-L, frozen).
2. **Split** each latent z into canonical coordinates q and momenta p (512-d each).
3. **Learn** a Hamiltonian H(q, p) → R as a small MLP (~0.9M params).
4. **Integrate** dynamics via leapfrog with a learnable timestep dt. This preserves the symplectic 2-form exactly.

## Results

On synthetic 2-ball elastic collision videos:

| Model | MSE | Energy Drift | Params |
|---|---|---|---|
| **H-JEPA (full)** | **best** | **best** | 0.9M |
| HNN (no energy loss) | +ablation | worse | 0.9M |
| HNN (Euler integrator) | +ablation | worse | 0.9M |
| Baseline MLP | 1.9x higher | higher | 1.3M |

The symplectic integrator and the energy conservation loss both matter. Full numbers and ablation details are in the paper.

## Repository contents

```
hjepanotebook.ipynb          # Full pipeline: data generation, V-JEPA 2 extraction,
                             #   HNN training, ablations, evaluation, and plots
hjepapaper/
  iclr2025_conference.pdf    # Paper
```

## Running the notebook

Built for **Google Colab with a GPU runtime** (T4 or better). Here's what it does:

1. Generates 50 synthetic collision videos
2. Extracts V-JEPA 2 latents (needs ~3GB VRAM)
3. Trains the HNN and ablation models (50 epochs each)
4. Produces all figures and the summary table

Dependencies get installed in the first cell, but for reference:
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
