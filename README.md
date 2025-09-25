# EmoLat: Text-driven Image Sentiment Transfer via Emotion Latent Space

## Introduction
We proposes a novel emotional latent space (EmoLat), whose core function is to achieve fine-grained, text-driven image sentiment transfer by modeling cross-modal correlations between textual semantics and visual emotional features.

## Dataset

As an expanded and enhanced version of existing emotion datasets (e.g., EmoSet), Emospace Set addresses the limitations of current datasets in terms of data scale, emotion category granularity, and emotion annotation completeness. Specifically, it remedies key shortcomings of EmoSet: lack of fine-grained attribute annotations, overly generalized emotion labels (only 8 categories, failing to match the diversity of real-world emotions), and limited emotional expressiveness in visual features. This enables it to meet the needs of in-depth research in the field of image emotion transfer.

The final Emospace Set contains 118,100 images, with each image having up to 14 object-attribute pairs. The entire dataset includes 1,953 unique attributes and 27,625 distinct object-attribute pairs, providing abundant semantic and emotional information.

You can obtain the dataset through the following links:

## Pretrained weights

You can download the pretrained weights files from these links:


## Training

1. Modify the Weight Path.

2. Run the following code for training

   ```bash
   python train.py
   ```
