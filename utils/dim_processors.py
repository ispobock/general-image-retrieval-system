from abc import abstractmethod
from typing import Dict
from .base import ModuleBase

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as SKPCA

class DimProcessorBase(ModuleBase):
    """
    The base class of dimension processor.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        ModuleBase.__init__(self, hps)

    @abstractmethod
    def __call__(self, fea: np.ndarray) -> np.ndarray:
        pass

class L2Normalize(DimProcessorBase):
    """
    L2 normalize the features.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(L2Normalize, self).__init__(hps)

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        return normalize(fea, norm="l2")
    

class PCA(DimProcessorBase):
    """
    Do the PCA transformation for dimension reduction.

    Hyper-Params:
        proj_dim (int): the dimension after reduction. If it is 0, then no reduction will be done.
        whiten (bool): whether do whiten.
        train_fea_dir (str): the path of features for training PCA.
        l2 (bool): whether do l2-normalization for the training features.
    """
    default_hyper_params = {
        "proj_dim": 0,
        "whiten": False,
        "l2": True,
        "random_state": 42
    }

    def __init__(self, train_fea, hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(PCA, self).__init__(hps)

        self.pca = SKPCA(n_components=self._hyper_params["proj_dim"], whiten=self._hyper_params["whiten"], random_state=self._hyper_params["random_state"])
        self._train(train_fea)

    def _train(self, train_fea) -> None:
        """
        Train the PCA.

        Args:
            fea_dir (str): the path of features for training PCA.
        """
        if self._hyper_params["l2"]:
            train_fea = normalize(train_fea, norm="l2")
        self.pca.fit(train_fea)

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        ori_fea = fea
        proj_fea = self.pca.transform(ori_fea)
        return proj_fea


DIMPROCESSORS = {
    'L2Normalize': L2Normalize,
    'PCA': PCA
}

def build_dim_processor(name, **kwargs):
    if name not in DIMPROCESSORS.keys():
        raise KeyError("Invalid dim_processor, got '{}', but expected to be one of {}".format(name, DIMPROCESSORS.keys()))
    dim_processor = DIMPROCESSORS[name](**kwargs)
    return dim_processor