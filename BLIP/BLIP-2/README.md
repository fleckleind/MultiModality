# BLIP-2
[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q/li23q.pdf) 

BLIP-2: a generic and efficient pre-training strategy bootstrapping vision-language pre-trianing from off-the-shelf frozen pre-trained image encoders and frozen large language models, bridging the modality gap with a lightweight Querying Transformer pretrained in two stages.

## Q-Former Architecture
Q-Former: a trainable module to bridge the gap between a frozen image encoder and a frozen LLM, including:
1. Image Transformer: interact with the frozen image encoder for visual feature extraction in cross-attention layer between self-attention (SA) layer and feed forward (FFN) layer.
2. Text Transformer: function as both text encoder and text decoder, sharing same parameters in SA layer with image transformer.

A set number of learnable query embeddings is created as input to the image transformer, interacting with each other through SA, frozen image features through CA, and text through shared SA in text transformer. Q-Former is initialized with the pre-trained weights of $BERT_{base}$, whereas the CA layers are randomly initialized.

## Vision-Language Representation Learning
Q-Former is jointly optimized by three pre-training objectives sharing same input format and model parameters with different attention masking strategy between queries and text to control their interaction.

### ITC: Image-Text Contrastive Learning
ITC: align the query representation $Z$ from image transformer with the text representation $t$ from the text transformer, where $t$ is the output embedding of the [CLS] token, and select the highest pairwise similarity between each query and $t$ as the image-text similarity.  

Uni-modal Self-Attention Mask: avoid ingotmation leak, where the queries and text are not allowed to see each other.

### ITG: Image-grounded Text Generation
ITG: generate texts with given input images as condition, and replace the [CLS] with [DEC] as the first text token to signal the decoding task.

Multi-modal Causal Self-Attention Mask: queries can attend to each other but not the text tokens, and each text token can attend to all queries and its previous text tokens.

### ITM: Image-Text Matching
ITM: a binary classification task, learn fine-grained alignment between image and text representation, with output query embeddings $Z$ capturing multimodal information.

Bi-direction Self-Attention Mask: all queries and texts can attend to each other, with all position unmasked.

## Vision-to-Language Generative Learning
To linearly project the output query embeddings $Z$ to the same dimension as the text embedding of the LLM, a fully-connected layer is used. Then the projected query emebddings are prepended to the input text embeddings as soft visual prompts, reducing the burden of the LLM to learn vision-language alignment and mitigating the catastrophic forgetting problem.

Different training strategies for two types of LLMs:
1. Decoder-based LLMs: pre-train with the language modeling loss, where frozen LLM is tasked to generate the text conditioned on the visual representation from Q-Former.
2. Encoder-Decoder-based LLMs: pre-train with the preflix language modeling loss, where a text is splited into two parts, with prefix text concatenated with the visual representation as input, and suffix text as the generation target.
