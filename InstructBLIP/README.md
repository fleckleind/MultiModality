# InstructBLIP
[InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/pdf/2305.06500)

InstructBLIP: introduce an instruction-aware Query Transformer to extract informative features tailored to the given instruction.

## Insturction-aware Visual Feature Extraction
Q-Former: extract visual features from the frozen image encoder, with a set of $K$ learnable query embeddings input to interact with the output of image encoder output via cross-attention.

Instruction-aware Q-Former: take instruction text tokens as additional input, interacting with the query embeddings through self-attention layers to encourage the extraction of task-relevant image features.

## Implementation Detais
Instruction-tuned LLMs: FlanT5, based on encoder-decoder Transformer T5; and Vicuna, decoder-only Transformer instruction-tuned from LLaMA.

## Training Dataset
Held-in and Held-out Datasets: with two types of held-out daya, tasks presenting in the held-in cluster and unseen tasks during training.

Tasks: image captioning (w or w/o reading comprehension, visual reasoning), image question answering, knowledge-grounded image question answering, image question answering with reading comprehension, image question generation (adapted from the QA datasets), video question answering, visual conversational question answering, image classification, and LLaVA-Instruct-150K.

Balance Dataset Sampling Strategy: probabilities proportional to the square root of their sizes, or the numbers of training samples. Given $D$ datasets with sizes $\{S_1, S_2, \ldots, S_D\}$, the probability of the daya sample being selected from dataset $d$ during training is:
```math
p_d=\frac{\sqrt{S_d}}{\sum_{i=1}^D \sqrt{S_i}}
```
