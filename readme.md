# ConvMAE pytorch 

## Adding FCMAE

The Co-Design Masked Autoencoder (CoMAE) framework is a CNN-based self-supervised learning method. It randomly masks regions of the input image and trains the model to reconstruct the masked parts. This forces the network to learn both global and local image features, improving generalization.

Compared to traditional Masked Autoencoder (MAE) frameworks, CoMAE offers two main advantages:

- It uses a fully convolutional architecture instead of fully connected layers to generate masks and reconstruct images, reducing parameter count and computation while preserving spatial information.
- It adopts a multi-scale masking strategy rather than fixed-size masks, improving the model’s ability to capture features at different spatial scales.
[FCMAE(fully convolutional masked autoencoder framework)](https://github.com/Jacky-Android/convnext-v2-pytorch/blob/main/fcmae_model.py)

![image](https://github.com/Jacky-Android/convnext-v2-pytorch/assets/55181594/cb3f3944-c0b6-4bba-86b3-d38f75fadcc6)

### torchinfo Output Code
```python
from fcmae_model import convnextv2_pico
from torchinfo import summary
import torch

#use pico
models = convnextv2_pico().cuda()
x = torch.randn([1,3,224,224]).cuda()
print(models(x)[0],models(x)[1].shape,models(x)[2].shape)
out = summary(models, (1, 3, 224,224))
```
### Model torchinfo Output
```python
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
CoMAE                                         --                        512
├─SparseConvNeXtV2: 1-1                       [1, 512, 7, 7]            --
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-1                   [1, 64, 56, 56]           3,264
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-2                   [1, 64, 56, 56]           73,856
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-3                   [1, 128, 28, 28]          33,024
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-4                   [1, 128, 28, 28]          278,784
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-5                   [1, 256, 14, 14]          131,584
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-6                   [1, 256, 14, 14]          3,245,568
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-7                   [1, 512, 7, 7]            525,312
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-8                   [1, 512, 7, 7]            4,260,864
├─Conv2d: 1-2                                 [1, 512, 7, 7]            262,656
├─Sequential: 1-3                             [1, 512, 7, 7]            --
│    └─Block: 2-9                             [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-9                       [1, 512, 7, 7]            25,600
│    │    └─LayerNorm: 3-10                   [1, 7, 7, 512]            1,024
│    │    └─Linear: 3-11                      [1, 7, 7, 2048]           1,050,624
│    │    └─GELU: 3-12                        [1, 7, 7, 2048]           --
│    │    └─GRN: 3-13                         [1, 7, 7, 2048]           4,096
│    │    └─Linear: 3-14                      [1, 7, 7, 512]            1,049,088
│    │    └─Identity: 3-15                    [1, 512, 7, 7]            --
├─Conv2d: 1-4                                 [1, 3072, 7, 7]           1,575,936
===============================================================================================
Total params: 12,521,792
Trainable params: 12,521,792
Non-trainable params: 0
Total mult-adds (M): 235.88
===============================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 94.93
Params size (MB): 50.09
Estimated Total Size (MB): 145.62
===============================================================================================
```
## Code Instructions


1. Download the dataset — the code uses the flower classification dataset by default: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz. If that is unavailable, use the Kaggle mirror: https://www.kaggle.com/datasets/l3llff/flowers.

2. In `train.py` set `--data-path` to the absolute path of the extracted `flower_photos` folder.

3. Download pretrained weights. Each model in model.py includes a download URL — download the pretrained file that matches the model you will use.

4. In train.py set --weights to the path of the downloaded pretrained weights.

5. With --data-path and --weights configured, run train.py to start training. The training process will automatically generate class_indices.json.

6. In predict.py import the same model used for training and set model_weight_path to the trained model weights (by default saved in the weights folder).
Note: Set num_classes to the number of classes in your dataset.

5. With `--data-path` and `--weights` configured, run `train.py` to start training. The training run will automatically generate `class_indices.json`.

6. In `predict.py` import the same model used for training and set `model_weight_path` to the trained model weights (by default saved in the `weights` folder).

7. In `predict.py` set `img_path` to the absolute path of the folder containing the images you want to predict. The script will output `results.csv`.

8. With `model_weight_path` and `img_path` set, run `predict.py` to perform predictions.

9. To use your own dataset, follow the flower dataset folder structure (one folder per class) and set `num_classes` in the training and prediction scripts to your dataset’s number of classes.
### ImageNet-1K CoMAE pre-trained weights (*self-supervised*)
| name | resolution | #params | model |
|:---:|:---:|:---:|:---:|
| ConvNeXt V2-B | 224x224 | 89M  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt) |


### ImageNet-1K fine-tuned models
| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt V2-B | 224x224 | 84.9 | 89M   | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt) |


### ImageNet-22K fine-tuned models
| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:| :---:|
| ConvNeXt V2-B | 224x224 | 86.8 | 89M   | 15.4G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt)|
| ConvNeXt V2-B | 384x384 | 87.7 | 89M   | 45.2G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt)|


