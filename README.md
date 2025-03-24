# Bird Classification
**Classifier & dataset for common bird species in China**

## Overview (Ver 1.0)

This project aims to classify common bird species in China (373 species) with a CNN. We also provide the image dataset (220k) for training and testing. Besides, we provide a simple GUI for easy classification.

--------------------------------------------------

该项目使用了卷积神经网络，以实现对中国常见鸟类（共373种）的图像分类识别。我们还提供了所使用的图像数据集（22万）。此外，我们提供了一个简单的GUI，方便用户使用。

## Image Dataset

The image dataset is collected from iNaturalist. It contains ~ 220K images of 373 species, and 80% are used for training. Each species has at least 50 iamges. About details, please refer to ``species_list.csv``, and [Note.md](./birdData/Note.md).

## Model Details

The model is a CNN network, which imitates the ConvNeXt architecture. Its input should be a 224x224 RGB image.

The latest version has ~ 12M parameters. Its accuracy on test set reaches **59.36% (top1) / 72.58% (top3)**.

## Usage

- **Dataset**: The dataset is in ``birdData`` folder. Download it and unzip the packages, and then you can use it to train your own model.
- **Training**: Run ``_train.py`` to train the model, and run ``_test.py`` to test trained model. Pretrained model is available in ``trained`` folder.
- **Classification GUI**: Run ``BirdAPP.py``, and then you can easily classify your own image with our GUI. You can simply drag and drop your image to the window, or select your image from the file dialog.

--------------------------------------------------

- **数据集**: 数据集在``birdData``文件夹中。下载并解压后，您可以使用它来训练自己的模型。
- **训练**: 运行``_train.py``来训练模型，运行``_test.py``来测试训练好的模型。预训练模型在``trained``文件夹中。
- **分类GUI**: 运行``BirdAPP.py``，然后您可以轻松地使用我们的GUI来识别自己的图像。您可以直接将图像拖放到窗口中，或者从文件对话框中选择图像。

--------------------------------------------------

![Example](./Example.png)

## Requirements

- Python 3.12
- Torch
- Torchvision
- Numpy
- PyQt5

## License

This project is licensed under the MIT License.

## Acknowledgments

The image dataset is exported from iNaturalist (https://www.inaturalist.org) on 2025/03.

Thanks for the help from open source community.