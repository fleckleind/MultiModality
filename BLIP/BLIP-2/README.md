# BLIP-2
[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q/li23q.pdf) 

BLIP-2: a generic and efficient pre-training strategy bootstrapping vision-language pre-trianing from off-the-shelf frozen pre-trained image encoders and frozen large language models, bridging the modality gap with a lightweight Querying Transformer pretrained in two stages.

## Q-Former Architecture
Q-Former: a trainable module to bridge the gap between a frozen image encoder and a frozen LLM, including:
1. Image Transformer: interact with the frozen image encoder for visual feature extraction in cross-attention layer between self-attention (SA) layer and feed forward (FFN) layer.
2. Text Transformer: function as both text encoder and text decoder, sharing same parameters in SA layer with image transformer.

A set number of learnable query embeddings is created as input to the image transformer, interacting with each other through SA, frozen image features through CA, and text through shared SA in text transformer.

## Vision-Language Representation Learning

## Vision-to-Language Generative Learning
