# BLIP

BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

BLIP: trains MED (Multi-modal mixture of Encoder-Decoder) to simultaneously pay attention to understanding- and generation-based tasks in Vision-Language Pre-training (VLP), and uses modules in MED to clear low-quality text-image data.

## MED

Multi-modal mixture of Encoder-Decoder (MED): image-text comparative learning, image-text matching, and image conditional language modeling.
1. Image Encoder: ViT, with patch tokenizer to obtain image embedding, and additional \[class\] token to represent global image features.
2. Text Encoder: BERT extracting text features for comparative learning, with \[class\] token attached to text for sentence summarization.
3. Image-Grounded Text Encoder: push image embedding from image encoder, and text features captured via bilateral self-attention into cross-attention to get binary/matching classification, with \[Encoder\] token for image-text joint characterization.
4. Image-Grounded Text Decoder: push image embedding from image encoder, and text features captured via causal self-attention into cross-attention to generate text/caption, with \[Decoder\] token as the beginning and end of synthetic results.
5. Notes: text encoder, image-grounded text encoder, and image-grounded text decoder share the same parameters in feed forward layers.

Pre-training Ambitions of BLIP: 
1. Image-Text Contrastive Loss (ITC): image encoder and text encoder, to align feature spaces of image and text.
2. Image-Text Matching Loss (ITM): image encoder and image-grounded text encoder, predicting (image, text) is positive/negative pair, to learn the joint characterization and align the fine-grained between image and text.
3. Language Modeling Loss (LM): image encoder and image-grounded text decoder, generating text description via auto-regressive manner, and transferring visual information into cohesive caption.


### ALBEF

Momentum Encoder: generate fake label to aid model training.  

Hard Negative Mining: 



##  CapFilt

 CapFilt: efficiently leverage noisy image-text data $(I_w, T_w)$ collected from Internet, including:
 1. Captioner: image-grounded text decoder fine-tuned on dataset $COCO$, given image $I_w$ and generate caption $T_s$.
 2. Filter: image-grounded text encoder fine-tuned on dataset $COCO$ with loss function $ITC$ and $ITM$, filter out noisy (not matching) image-text pairs, equaling to delete original text $T_w$ or synthetic text $T_s$ if $ITM$ head predicts not matching.
 3. New Dataset: combine filtered image-text pairs with manually denoted pairs and get new dataset to pre-train new BLIP.


