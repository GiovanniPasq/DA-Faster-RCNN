# Detectron2 implementation of DA-Faster-RCNN

This is the implementation of CVPR 2018 work 'Domain Adaptive Faster R-CNN for Object Detection in the Wild'. The aim is to improve the cross-domain robustness of object detection, in the screnario where training and test data are drawn from different distributions. The original paper can be found [here](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf)

## DA-Faster R-CNN architecture
<center><img src='DA-Faster-RCNN.png' width=100%/></center>

## Installation
You can use this repo following one of these three methods:<br>
NB: Detectron2 0.6 is required, installing other versions this code will not work.

### Google Colab
Load and run the ```DA-Faster-RCNN.ipynb``` on Google Colab following the instructions inside the notebook.

### Detectron 2 on your PC
Follow the official guide to install [Detectron2 0.6](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

### Detectron2 via Dockerfile
Follow the official guide to install [Detectron2 0.6](https://github.com/facebookresearch/detectron2/blob/main/docker/README.md)

## Dataset
You can find at the following links two datasets for Unsupervised Domain Adaptation for Object Detection:
[Cityscapes-Foggy Cityscapes](https://github.com/fpv-iplab/Cityscapes-FoggyCityscapes)<br>
[UDA-CH](https://github.com/fpv-iplab/DA-RetinaNet#dataset)

## Contributing
This repo is actively developed. Any contribution in the form of a suggestion, bug report or pull request, is well accepted 😊<br>
Please leave a star ⭐ if you use this repository for your project.

## Related Work
[DA-RetinaNet](https://github.com/fpv-iplab/DA-RetinaNet)<br>
[STMDA-RetinaNet](https://github.com/fpv-iplab/STMDA-RetinaNet)<br>
