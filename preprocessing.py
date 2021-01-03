import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)
import pickle

from dataloader import ImageDataset
from models import build_model
from utils import build_aggregator, build_dim_processor, build_feature_enhancer
from settings import MODEL, AGGREGATOR, DIM_PROCESSORS, FEATURE_ENHANCER, FEATURE_DIM, USE_GPU, GPU_DEVICES, EXTRACT_GALLERY_FEATURE, GALLERY_PATH
import torch.backends.cudnn as cudnn

from IPython import embed

USE_GPU = torch.cuda.is_available() and USE_GPU

def preprocess():
    model = build_model(MODEL)
    if USE_GPU:
        logging.info("Currently using GPU {}".format(GPU_DEVICES))
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_DEVICES
        cudnn.benchmark = True
        model.cuda()
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    aggregator = build_aggregator(AGGREGATOR)

    if EXTRACT_GALLERY_FEATURE:
        gallery_fea, img_ids = extract_gallery_feature(transform=transform, model=model, aggregator=aggregator)
    else:
        gallery_fea, img_ids = read_gallery_feature()

    dim_processors = []
    for proc in DIM_PROCESSORS:
        if proc == 'L2Normalize':
            dim_processors.append(build_dim_processor('L2Normalize'))
        elif proc == 'PCA':
            dim_processors.append(build_dim_processor('PCA', train_fea=gallery_fea, hps={'proj_dim':FEATURE_DIM}))

    for proc in dim_processors:
        gallery_fea = proc(gallery_fea)

    gallery_fea = torch.Tensor(gallery_fea)

    if FEATURE_ENHANCER is not None:
        feature_enhancer = build_feature_enhancer(FEATURE_ENHANCER)
        gallery_fea = feature_enhancer(gallery_fea)

    return gallery_fea, img_ids, model, aggregator, dim_processors, transform


def extract_gallery_feature(transform, model, aggregator):
    imgloader = DataLoader(
        ImageDataset(dataset_dir=GALLERY_PATH, transform=transform),
        batch_size=32,
        shuffle=False
    )

    feature_map  = {}
    def hook(module, input, output):
        feature_map['pool5'] = output.cpu().data

    handle = model.layer4.register_forward_hook(hook)

    gallery_fea = None
    img_ids = []

    for batch_idx, (img, img_id) in enumerate(imgloader):
        if USE_GPU:
            img = img.cuda()
        model(img)
        temp = aggregator(feature_map)
        if gallery_fea is None:
            gallery_fea = temp['pool5_'+AGGREGATOR].cpu().data
        else:
            gallery_fea = torch.cat((gallery_fea, temp['pool5_'+AGGREGATOR].cpu().data), 0)
        img_ids.extend(list(img_id))
    
    handle.remove()
    img_ids = np.array(img_ids)

    save_gallery_feature(gallery_fea, img_ids)
    
    return gallery_fea, img_ids

def save_gallery_feature(gallery_fea, img_ids, feature_path='gallery_feature.pkl'):
    dic = {
        'gallery_fea': gallery_fea,
        'img_ids': img_ids
    }
    with open(feature_path, "wb") as f:
        pickle.dump(dic, f)

def read_gallery_feature(feature_path='gallery_feature.pkl'):
    if not os.path.exists(feature_path):
        raise IOError("{} does not exist".format(feature_path))

    with open(feature_path, "rb") as f:
        dic = pickle.load(f)
    gallery_fea = dic['gallery_fea']
    img_ids = dic['img_ids']
    return gallery_fea, img_ids