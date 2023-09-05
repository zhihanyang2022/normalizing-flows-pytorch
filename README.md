# VI with normalizing flows

This repo documents my attempt at reproducing "Variational Inference with Normalizing Flows" in PyTorch. 

## 2023/9/5

I reproduced Section 6.1 using **planar flows**. 

- I used `torch.autograd.functional.jacobian` instead of Eq. 12 to compute the jacobian of the transformation; this is less efficient but (1) it doesn't matter because I'm using a small `D=2` and (2) it's easier to debug.
- I found that I didn't have to enforce invertibility as discussed in Section A.1; during training, the dot product between `w` and `u` was always greater than -1. Explicitly enforced this constraint in my code led to bad results (i.e., the learned density didn't resemble the true density). Maybe I was doing it wrong...

My questions:

- How did the authors evaluate the density at each point in Figure 3b without having access to the original `z0`? Really interesting conversation [here](https://groups.google.com/a/tensorflow.org/g/tfprobability/c/KouBOt9HQa8).
- Potential functions 2, 3 and 4 extend indefinitely? Did the authors set some cutoff value?

Anyway, here are some plots (with 100 layers of planar flows):

| Energy | True density (unnorm.) | Learned density (emp.) | After nth layer |
| :-: | :-: | :-: | :-: |
| U1  | Content Cell  | placeholder | placeholder |
| U2  | Content Cell  | placeholder | placeholder |
| U3  | Content Cell  | placeholder | placeholder |
| U4  | <img alt="image" src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/8a79a614-4278-412e-93ae-4b7692f2097d"> | <img alt="image" src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/af109a70-85cd-47d9-940b-1b4a6fbe9d46"> | <img alt="image" src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/1c757df3-f9e3-4d76-b9f1-01f67ca3aada"> |
