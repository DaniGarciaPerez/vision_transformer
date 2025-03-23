## <p align="center"> **Vision Transformers (ViT) from scratch** 👁️ </p>


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Repo Status](https://img.shields.io/badge/Repo%20Status-In%20Progress-red.svg)](https://github.com/DaniGarciaPerez/vision_transformer)

Hey, thank you for passing by! 

This is a repo to explore the implementation from scratch of the Vision Transformer architecture based on the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". Alexey Dosovitskiy et al. (2021)](https://arxiv.org/pdf/2010.11929) using PyTorch. I tried to keep everything as much similar to the paper as possible. Regarding the last layer of the network (the MLP layer used for classification), this implementation relies on Global Average Pooling (GAP) rather than the CLS token (similar to [BERT](https://arxiv.org/pdf/1810.04805)) based implementation used in the original paper, although they have a similar performance if trained with the right hyperparameters.

## **Content**

1. [Structure of the Project](#structure-of-the-project)
2. [Environment and Dependencies](#environment-and-dependencies)
3. [Setup and Use](#setup-and-use)
4. [Supported Datasets](#supported-datasets)

## **Structure of the project**

```bash
.
├── README.md
├── LICENSE
├──.gitignore
├── parameters.env
├── requirements.txt
└── src
    ├── main.py
    ├── encoder_block
    │   ├── encoder.py
    │   ├── mlp_encoder.py
    │   ├── multihead_attention.py
    │   └── scaled_dot_product.py
    ├── model_components
    │   ├── mlp_head.py
    │   ├── patch_projection.py
    │   ├── positional_encoding.py
    │   └── vit.py
    └── training_utils
        ├── dataset_loader.py
        ├── optimizer.py
        └── train.py
```

## **Environment and Dependencies**

This project was developed using:

- python 3.12.
- python modules as described in requirements.txt

## **Supported Datasets**

- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://www.image-net.org/)
- [FashionMnist](https://en.wikipedia.org/wiki/Fashion_MNIST)
  
## **Setup and Use**

In order to train a model using any of the dataset listed above:

1. Install Dependencies: ```pip install -r requirements.txt```
2. Configure Hyperparameters: Modify parameters.env as needed.
3. Execute ```python .\src\main.py``` from the project root.

