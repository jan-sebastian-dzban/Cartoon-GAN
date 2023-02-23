# Cartoon-GAN

## Description
My approach on the problem of transferring
the style of cartoon images to real-life photographic images by
implementing previous work done by CartoonGAN. We trained
a Generative Adversial Network(GAN) on over 60 000 images
from works by Hayao Miyazaki at Studio Ghibli.  

To the people asking for the dataset, im sorry but as the material is copyright protected i cannot share the dataset.


## Weights
Weights for the presented models can be found [here](https://drive.google.com/drive/folders/1d_GsZncTGmMdYht0oUWG9pqvV4UqF_kM?usp=sharing)


## Training

All training code can be found in `experiment.ipynb`

## Predict

Predict by running `predict.py`.

Example:

```
python predict.py -i C:/folder/input_image.png -o ./output_folder/output_image.png
```

Predictions can be made on images, videos or a folder of images/videos.

## Demonstration

| Image # | Original | CartoonGAN | GANILLA | Our implementation |
|:-------:|----------|------------|---------|--------------------|
|1| ![Original_1](https://i.imgur.com/7j3ysv0.png) | ![CartoonGAN_1](https://i.imgur.com/4g9VgjJ.jpg) | ![GANILLA_1](https://i.imgur.com/dAuJtfd.png) | ![Ours_1](https://i.imgur.com/wSFvpqm.png) |
|2| ![Original_2](https://i.imgur.com/A3nIuQd.png) | ![CartoonGAN_2](https://i.imgur.com/pzLGkR0.jpg) | ![GANILLA_2](https://i.imgur.com/SF0o9Ta.png) | ![Ours_2](https://i.imgur.com/Eaqmu7g.png) |
|3| ![Original_3](https://i.imgur.com/kad7Q9k.png) | ![CartoonGAN_3](https://i.imgur.com/twlJb0R.jpg) | ![GANILLA_3](https://i.imgur.com/MSLtpZv.png) | ![Ours_3](https://i.imgur.com/5haiEKj.png) |


