# General Image Retrieval System (Backend)
This is a general image retrieval system demo. We implements some general function modules, such as feature extraction (Resnet50 + Hook), feature aggregation (SCDA), dimension processing (L2Normalize, PCA), distance calculation (cosine, L2) and k-nearest neighbor retrieval. In the implementation process, we refer to [Pyretri](https://github.com/PyRetri/PyRetri) open source framework.

## Framework
![image](https://github.com/ispobock/general-image-retrieval-system/blob/master/imgs/framework.svg)

## Requirements
- Python
- PyTorch
- Flask
- Flask RESTful
- Gunicorn
- numpy
- sk-learn