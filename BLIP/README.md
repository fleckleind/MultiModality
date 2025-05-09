# BLIP
[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n/li22n.pdf)  

BLIP: a new Vision-Language Pre-training (VLP) framework flexibly transfers to both vision-language understanding and generation tasks, and effectively utilizes the noisy web data by bootstrapping the captions, with a captioner generating captions and a filter removing noisy ones.

## MED: Multimodal mixture of Encoder-Decoder
MED can operate either as a unimodal encoder, an image-grounded text encoder, or an image-grounded text decoder, with jointly pre-trained with three vision-language objectives including image-text contrastive learning, image-text matching, and image-conditioned language modeling.  

### Model Architecture
Image Encoder: visual transformer (ViT), dividies an input image into patches and encodes them as a sequence of embeddings, with an additional [CLS] token to represent the flobal image feature.  

Unimodal Encoder: BERT (bi-direction self-attention and feed forward layer), seperately encodes image and text, with a [CLS] token added to the beginning of the text input to summarize the sentence.

Image-Grounded Text Encoder: insert additional cross-attention (CA) layer between shared BiSA and FFN in unimodal encoder to inject visual information, with a task-specific [Encode] token for multimodal representation of the image-text pair in the ouput embedding.

Image-Grounded Text Decoder: replaces BiSA with casual self-attention, with shared CA and FFN in image-grounded text encoder, with [Decode] token to signal the beginning and end of a sequence.

### Pre-training Objectives
#### Image-Text Contrastive Loss
Image-Text Contrastive Loss (ITC): understanding-based objectives, to align the feature space of the image encoder and unimodal encoder by encouraging positive image-text pairs to have similar representation in contrast to the negative pairs. ITC is defined as the cross-entropy $H$ between similarity $p$ and ground-truth $y$, 
```math
L_{ITC}=\frac{1}{2}E_{(I,T)\sim D}[H(y^{i2t}(I), p^{i2t}(I)) + H(y^{t2i}(T), p^{t2i}(T))]
```
With similarity function as $s(I,T)=g_v(v_{cls})^T g_w^\prime(w_{cls}^\prime)$ and $s(T,I)=g_w(w_{cls})^T g_v^\prime(v_{cls}^\prime)$, the softmax-normalized image-to-text and text-to-image similarity is calulated as follows, 
```math
p_m^{i2t}(I)=\frac{exp(s(I,T_m)/\tau)}{\sum_{m=1}^M exp(s(I,T_m)/\tau)}, \quad
p_m^{t2i}(T)=\frac{exp(s(T,I_m)/\tau)}{\sum_{m=1}^M exp(s(T,I_m)/\tau)}
```

#### Image-Text Matching Loss
Image-Text Matching Loss (ITM): a binary classification task, predicting whether an image-text pair is positive (matched) or negative (unmatched), to learn image-text multi-modal representation capturing the fine-grained alignment between vision and language.
```math
L_{ITM}=E_{(I,T)\sim D} H(y^{itm}, p^{itm}(I,T))
```
where two-class probability $p^{itm}(I,T)$ is calculated by [Encode] token with a FC layer followed by softmax, y^{itm} is a 2-dimensional one-hot vector representing the ground-truth label. 

#### Language Modeling Loss
Language Modeling Loss (LM): to generate textual descriptions given an image, optimizes model with cross entropy loss (label smoothing=0.1) to maximize the likelihood of the text in an autoregressive manner.
```math
L_{LM}=E_{(I,T)\sim D} H(y^{msk}, p^{msk}(I,T))
```
where $y^{msk}$ is a one-hot vocabulary distribution where the ground-truth token has a probability of 1, and $p^{msk}(I,T)$ denotes the predicted probability for [Decode] token.

## CapFilt: Captioning and Filtering
CapFilt: a new dataset boostrapping method for learning from noisy image-text pairs, including a captioner to produce synthetic captions given web images, and a filter to remove noisy captions from both the original web texts and the synthetic texts.

The captioner is an image-grounded text decoder, fine-tuned with the LM objective to decode texts given images. Given the web images $I_w$, the captioner generates synthetic captions $T_s$ with one caption per image. 
The filter is an image-grounded text encoder, fine-tuned with the ITC and ITM objectives to learn whether a text matches an image. The filter removes noisy texts in both the original web texts $T_w$ and the synthetic texts $T_s$, where a text is considered to be noisy if the ITM head predicts it as unmatched to the image.
Finally, the filtered image-text pairs are combined with the human-annotated pairs to form a new dataset for new model pre-training.

## Reference
[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://proceedings.neurips.cc/paper_files/paper/2021/file/505259756244493872b7709a8a01b536-Paper.pdf)

