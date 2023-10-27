# Variational inference with normalizing flows

This is an unofficial implementation of "Variational inference with normalizing flows".

```bibtex
@inproceedings{rezende2015variational,
  title={Variational inference with normalizing flows},
  author={Rezende, Danilo and Mohamed, Shakir},
  booktitle={International conference on machine learning},
  pages={1530--1538},
  year={2015},
  organization={PMLR}
}
```

## Unconditional variational inference

See `core.py` and `nf_for_uncond_vi.ipynb`.

Legend:

- 1st image: unnormalized true density
- 2nd image: empirical learned density
- 3rd image: sampled points after passing through the $n$-th layer (each is colored by its distance to $\mu$ before passing through any layer)
- 4th image: estimated KL against number of gradient steps

Potential function $U_1$:

<p align="middle">
  <img src="imgs/U1_true_density.png" width="20%" />
  <img src="imgs/U1_learned_density.png" width="20%" /> 
  <img src="imgs/U1_samples_from_each_layer.png" width="26.667%" />
  <img src="imgs/U1_kl_over_time.png" width="20%" />
</p>

Potential function $U_2$ (tapered version):

<p align="middle">
  <img src="imgs/U2_true_density.png" width="20%" />
  <img src="imgs/U2_learned_density.png" width="20%" /> 
  <img src="imgs/U2_samples_from_each_layer.png" width="26.667%" />
  <img src="imgs/U2_kl_over_time.png" width="20%" />
</p>

Potential function $U_3$ (tapered version):

<p align="middle">
  <img src="imgs/U3_true_density.png" width="20%" />
  <img src="imgs/U3_learned_density.png" width="20%" /> 
  <img src="imgs/U3_samples_from_each_layer.png" width="26.667%" />
  <img src="imgs/U3_kl_over_time.png" width="20%" />
</p>

Potential function $U_4$ (tapered version):

<p align="middle">
  <img src="imgs/U4_true_density.png" width="20%" />
  <img src="imgs/U4_learned_density.png" width="20%" /> 
  <img src="imgs/U4_samples_from_each_layer.png" width="26.667%" />
  <img src="imgs/U4_kl_over_time.png" width="20%" />
</p>

A potential function I created:

<p align="middle">
  <img src="imgs/U8_true_density.png" width="20%" />
  <img src="imgs/U8_learned_density.png" width="20%" /> 
  <img src="imgs/U8_samples_from_each_layer.png" width="26.667%" />
  <img src="imgs/U8_kl_over_time.png" width="20%" />
</p>

## Conditional variational inference

See `core_vectorized.py` and `nf_for_cond_vi.ipynb`.

![conditional_vi](https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/04baca46-c548-4ba8-9ea0-9c3bcf872f9a)

## Training VAEs

See `nfvae_train.py` and `nfvae_plot_learning_curves.py`.

<img width="817" alt="image" src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/5d7a19bb-1707-41b6-bccf-ce11bc3553e5">
