# BLIP-3
[xGen-MM (BLIP-3): A Family of Open Large Multimodal Models](https://arxiv.org/pdf/2408.08872?)

BLIP-3: introduce a safety-tuned model with DPO, replace Q-Former layers with more scalable vision token sampler, and simplify the training process via the unification of the training objectives to a single loss at every training stage, aiming to mitigate harmful behaviors such as hallucinations and improve safety.

## Model Architecture
BLIP-3 framework consists of ViT, vision tken sampler (perceiver resampler) to downsample the image embeddings, and pre-trained LLM (phi3-mini), with input as free-form multimodal interleaved texts and vision tokens.

Perceiver Resampler: transfer image embedding into image token with fixed number via learnable queries. Perceive resampler has Transformer structure, concatenates image embedding and leanable queries, get $k, v$ from concatenated features, and $q$ from learnable queries in attention calculation. 

### Any-Resolution Vision Token Sampling
Dynamic High-Resolution ($i.e.$ Any-Resolution) Image Encoding Strategy: at the fine-tuning and post-training stages, preset templates to get most suitable resolution via following objection:
```math
arg\mathop{min}_t(wasted\_resolution), \quad t=1, 2, \ldots, N
```
where for $N$ templates 
```math
r_t = min(\frac{w_t}{w_{ori}}, \frac{h_t}{h_{ori}})
```
the calculation of the wasted resolutions are as follows,
```math
wasted\_resolution =w_t*h_t-min(w_{ori}*h_{ori}, INT(w_{ori}*r_t)*INT(h_{ori}*r_t)
```

Image Patch-Wise Encoding: split single image into multiple patches and encode them separately, preseving the resolution of the original image, then concatenate the encoded image patches with downsized original image providing global information.

## Training Paradigm
Traing paradigm includes four recipes:
1. Pre-Training: predict the next text token across the dataset mixture pre-train on.
2. Supervised Fine-Tuning (SFT): fine-tune pre-trained models on instruction-following examples, with any-resolution vision token sampling strategy for higher-resolution image understanding.
3. Interleaved Multi-Image Supervised Fine-tuning: second-stage instruction fine-tuning on a mixture of multi-image and single-image instructions-following samples.
4. Post-Training: two stages, direct preference optimization (DPO) for hallucination and safety fine-tuning to improve harmlessness.
