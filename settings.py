MODEL = 'resnet50'
AGGREGATOR = 'SCDA'
DIM_PROCESSORS = ['L2Normalize', 'PCA', 'L2Normalize']
FEATURE_ENHANCER = None
RE_RANKER = None
METRIC = 'Cosine_KNN'

FEATURE_DIM = 64
NUM_TO_RETRIEVE = 10

USE_GPU = False
GPU_DEVICES = '1,2'

EXTRACT_GALLERY_FEATURE = False
GALLERY_PATH = '/home/bk/general_sys/data/gallery/'