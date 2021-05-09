
# Reverse Image Search Engine

## Overview
This project's purpose is to build a reverse image search engine using ResNet-152, Milvus, and Flask. We first use PyTorch to re-train a pre-trained ResNet-152 model with trainable parameters only at 2 layers. After that, we use the resulting ResNet-152 model to encode the Stanford Dogs Dataset into vectors. Then, we use Milvus to create a collection and insert the vectors in the collection. Finally, we build a website interface using Flask.  

When an input image comes in, our application first uses the resulting ResNet-152 model to translate the image into a vector. Since all the vectors in the collection are normalized, the query vector is then used by the cosine similarity algorithm to find the most related vectors in the collection through Milvus. The returned vectors then correspond of the images in the Stanford Dogs Dataset, and the images are returned as the most related results.

## Data Source
1. Input data for the ResNet-152 model (unzip the file and put it in the **deep_learning** folder):
   https://drive.google.com/file/d/124Q-H1MKjFYib3cIAb9GMGfkF_Mg3LGv/view?usp=sharing
2. Images for displaying (unzip the file and put it in the root folder):
   https://drive.google.com/file/d/1nLP_rRrpNQE9KVfyCtAZY-Gp8RfM31SL/view?usp=sharing

&nbsp;

![Website Screeshot](https://github.com/KUANCHENGFU/Reverse-Image-Search-Engine/blob/main/static/screenshot.png)