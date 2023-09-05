# VI with normalizing flows

This repo documents my attempt at reproducing "Variational Inference with Normalizing Flows" in PyTorch. 

[2023/9/5] I reproduced Section 6.1 using **planar flows**. 
- I used `torch.autograd.functional.jacobian` instead of Eq. 12 to compute the jacobian of the transformation; this is less efficient but (1) it doesn't matter because I'm using a small D and (2) it's easier to debug.
- I found that I didn't have to enforce invertibility as discussed in Section A.1; during training, the dot product between `w` and `u` was always greater than -1. Explicitly enforced this constraint in my code led to bad results (i.e., the learned density didn't resemble the true density). Maybe I was doing it wrong...

Anyway, here are some plots:

