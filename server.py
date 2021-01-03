from flask import Flask, request
from flask_cors import CORS
import logging
logging.basicConfig(level=logging.INFO)

import os
import pickle

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn

from preprocessing import preprocess
from utils import build_metric, build_reranker
from settings import AGGREGATOR, RE_RANKER, NUM_TO_RETRIEVE, USE_GPU
USE_GPU = torch.cuda.is_available() and USE_GPU

# preprocessing
gallery_fea, img_ids, model, aggregator, dim_processors, transform = preprocess()

query_feature_map = {}
def query_hook(module, input, output):
    query_feature_map['pool5'] = output.cpu().data

handle = model.layer4.register_forward_hook(query_hook)

metric = build_metric('KNN')
if RE_RANKER is not None:
    re_ranker = build_reranker(RE_RANKER)

# application
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "hello"

@app.route("/inference", methods=['POST'])
def inference():
    query_img = Image.open(request.files.get('img')).convert('RGB')
    query_img = transform(query_img).unsqueeze(dim=0)

    if USE_GPU:
        query_img = query_img.cuda()

    # inference
    model(query_img)

    # aggregate features
    temp = aggregator(query_feature_map)
    query_fea = temp['pool5_'+AGGREGATOR].cpu().data

    # dimension processing
    for proc in dim_processors:
        query_fea = proc(query_fea)
    query_fea = torch.Tensor(query_fea)

    # KNN search
    dis, sorted_index = metric(query_fea, gallery_fea)

    # reranking
    if RE_RANKER is not None:
        sorted_index = re_ranker(query_fea, gallery_fea, dis=dis, sorted_index=sorted_index)

    result = {
        'img_ids': img_ids[sorted_index[0][:NUM_TO_RETRIEVE].numpy()].tolist(),
        'distance': dis[0].numpy()[sorted_index[0][:NUM_TO_RETRIEVE].numpy()].tolist()
    }
    
    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=7000)