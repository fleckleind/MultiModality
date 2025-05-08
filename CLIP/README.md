# CLIP
[Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)  

Contrastive Language-Image Pre-Training (CLIP): multi-modal model for training transferable visual model, with properties of zero-shot transfer, natural language supervision, and multi-modal learning.

## ConVIRT
[Contrastive Learning of Medical Visual Representations from Paired Images and Text](https://arxiv.org/pdf/2010.00747)  

ConVIRT: an altenative unsupervised strategy to learn visual representations by exploiting naturally occurring paired descriptive text, with input and ambition as follows:
1. Paired input $(x_v,x_u)$, with $x_v$ as images and $x_u$ as text sequence describing corresponding imaging information.
2. Learn a parameterized image encoder funciton $f_v$, mapping an image to a fixed-dimensional vector.

With image encoder $f_v$, text encoder $f_u$ and corresponding non-linear projection $g_v$ and $g_u$, the random view $\tilde{x}_v$ and $\tilde{x}_u$ are converted into $d$-dimensional vector representations $h_v$ and $h_u$ then $v$ and $u$.
```math
v=g_v(f_v(\tilde{x}_v)), \quad u=g_u(f_u(\tilde{x}_v))
```
In details, the image encoder is ResNet50, text encoder is BERT, non-linear projection with ReLU activation, and the random image transformations includes cropping, horizontal flipping, affine transformation, color jittering, and Gaussian blur.  

At training time, $(v_i, u_i)$ is used to denote the $i$=th pair, and objective of ConVIRT involves bi-direction image-to-text contrastive loss $l^{v\rightarrow u}$ and similar text-to-image contrastive loss $l_i^{u\rightarrow v}$, with $\langle v,u\rangle$ representing the cosine similarity $v^Tu/\lVert v\rVert\lVert u\rVert$, $\tau\in R^+$ as a temperature parameter, and scalar weight $\lambda\in\[0,1\]$.
```math
\begin{align}
l_i^{v\rightarrow u} &=-log\frac{e^{\langle v_i, u_i\rangle\tau}}{\sum_{k=1}^N e^{\langle v_i, u_k\rangle/\tau}} \\
l_i^{u\rightarrow v} &=-log\frac{e^{\langle u_i, v_i\rangle\tau}}{\sum_{k=1}^N e^{\langle u_i, v_k\rangle/\tau}} \\
L &=\frac{1}{N}\sum_{i=1}^N(\lambda l_i^{v\rightarrow u} + (1-\lambda) l_i^{u\rightarrow v})
\end{align}
```

## CLIP: Contrastive Pre-Training
CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples, while standard image models jointly train an image feature extractor and a linear classifier to predict some label.  

Given a batch of $N$ (image, text) pairs, CLIP is trained to predict which of the $N\times N$ possible (image, text) pairings across a batch acutally occured. CLIP then learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of th $N$ real pairs in the batch with minimizing the cossine similarity of the embeddings of the $N^2-N$ incorrect pairings.

Based on ConVIRT, CLIP mainly makes the following simplifications:
1. CLIP initializes the Image Encoder randomly, rather than using ImageNet.
2. CLIP only uses resize and squared crop in Image Transformation, and 





