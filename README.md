
# Reverse Image Search Engine

## Overview
This project's purpose is to build a reverse image search engine using ResNet-152, Milvus, and Flask. We first use PyTorch to re-train a pre-trained ResNet-152 model with trainable parameters only at 2 layers. After that, we use the resulting ResNet-152 model to encode Stanford Dogs Dataset into vectors. Then, we use Milvus to create a collection and insert the vectors in the collection. Finally, we build a website interface using Flask. When an input picture comes in, our application first uses resulting ResNet-152 model to translate the image into a vector. Since all the vectors are normalized, the query vector is then used by the cosine similarity algorithm to find the most related vectors through Milvus. The returned vectors then correspond of the images and are returned as the most related results.

![Website Screeshot](https://github.com/KUANCHENGFU/Reverse-Image-Search-Engine/blob/main/static/screenshot.png)
