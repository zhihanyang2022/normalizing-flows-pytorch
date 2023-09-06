# VI with normalizing flows

This repo documents my attempt at reproducing "Variational Inference with Normalizing Flows" in PyTorch. 

## 2023/9/5

I reproduced Section 6.1 using **planar flows**. 

- I used `torch.autograd.functional.jacobian` instead of Eq. 12 to compute the jacobian of the transformation; this is less efficient but (1) I'm only using $D=2$ and (2) it's easier to debug.
- I found that I didn't have to enforce invertibility as discussed in Section A.1 🤷; during training, the dot product between $\vec{w}$ and $\vec{u}$ was always greater than -1. Explicitly enforced this constraint in my code led to bad results (i.e., the learned density didn't resemble the true density). Maybe I was doing it wrong...

My thoughts/questions:

- How did the authors evaluate the density at each point in Figure 3b without having access to the original $\vec{z}_0$? Really interesting conversation [here](https://groups.google.com/a/tensorflow.org/g/tfprobability/c/KouBOt9HQa8). I think (supported by this link) it can't be done unless we invert the normalizing flow. This could be done because the transformation is invertible but then the question is whether the inverse has a closed-form solution (probably no). I believe that finding a good way to invert the transformation is **critical** because this allows to find the probability of something under $q_K$, which is the entire point of having an invertible transformation (e.g., in GAN, we didn't have to do this).
- Potential functions 2, 3 and 4 extend indefinitely? Did the authors set some cutoff value?

Hyper-parameters:

- 100 layers of planar flows
- 1000 samples from $q_K$ to estimate KL
- Adam with a learning rate of 2e-3
- 10000 gradient steps (less than 5 minutes)

Potential function $U_1$:

<p align="middle">
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/764bd8f6-69dc-4caa-8b44-5ca8f5cb79fc" width="25%" />
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/fb498327-f645-4e20-9429-0ccee857192c" width="25%" /> 
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/3e2dd21b-359a-4fe5-8989-641b9efe0297" width="33.33%" />
</p>

Potential function $U_2$ (tapered version):

<p align="middle">
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/764bd8f6-69dc-4caa-8b44-5ca8f5cb79fc" width="25%" />
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/fb498327-f645-4e20-9429-0ccee857192c" width="25%" /> 
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/3e2dd21b-359a-4fe5-8989-641b9efe0297" width="33.33%" />
</p>

Potential function $U_3$ (tapered version):

<p align="middle">
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/764bd8f6-69dc-4caa-8b44-5ca8f5cb79fc" width="25%" />
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/fb498327-f645-4e20-9429-0ccee857192c" width="25%" /> 
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/3e2dd21b-359a-4fe5-8989-641b9efe0297" width="33.33%" />
</p>

Potential function $U_4$ (tapered version):

<p align="middle">
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/764bd8f6-69dc-4caa-8b44-5ca8f5cb79fc" width="25%" />
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/fb498327-f645-4e20-9429-0ccee857192c" width="25%" /> 
  <img src="https://github.com/zhihanyang2022/vi-with-normalizing-flows/assets/43589364/3e2dd21b-359a-4fe5-8989-641b9efe0297" width="33.33%" />
</p>
