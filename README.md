# VI with normalizing flows

This repo documents my attempt at reproducing the paper "Variational Inference with Normalizing Flows" in PyTorch. I'm only trying to reproduce Section 6.1 using planar flows. In my experiments, I found that I didn't have to enforce invertibility as discussed in Section A.1; during training, the dot product between `w` and `u` was always greater than -1. Explicitly enforced this constraint in my code led to bad results (i.e., the learned density didn't resemble the true density).
