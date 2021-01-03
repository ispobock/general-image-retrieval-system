import torch
import queue

from typing import Dict, List

from abc import abstractmethod
from .base import ModuleBase

class AggregatorBase(ModuleBase):
    r"""
    The base class for feature aggregators.
    """
    default_hyper_params = dict()

    def __init__(self, hps):
        super(AggregatorBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, features):
        pass

class Crow(AggregatorBase):
    """
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    c.f. https://arxiv.org/pdf/1512.04065.pdf

    Hyper-Params
        spatial_a (float): hyper-parameter for calculating spatial weight.
        spatial_b (float): hyper-parameter for calculating spatial weight.
    """
    default_hyper_params = {
        "spatial_a": 2.0,
        "spatial_b": 2.0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(Crow, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        spatial_a = self._hyper_params["spatial_a"]
        spatial_b = self._hyper_params["spatial_b"]

        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                spatial_weight = fea.sum(dim=1, keepdims=True)
                z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdims=True)
                z = z ** (1.0 / spatial_a)
                spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

                c, w, h = fea.shape[1:]
                nonzeros = (fea!=0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
                channel_weight = torch.log(nonzeros.sum(dim=1, keepdims=True) / nonzeros)

                fea = fea * spatial_weight
                fea = fea.sum(dim=(2, 3))
                fea = fea * channel_weight

                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[Crow Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret


class GAP(AggregatorBase):
    """
    Global average pooling.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GAP, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                fea = fea.mean(dim=3).mean(dim=2)
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[GAP Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret

class GeM(AggregatorBase):
    """
    Generalized-mean pooling.
    c.f. https://pdfs.semanticscholar.org/a2ca/e0ed91d8a3298b3209fc7ea0a4248b914386.pdf

    Hyper-Params
        p (float): hyper-parameter for calculating generalized mean. If p = 1, GeM is equal to global average pooling, and
            if p = +infinity, GeM is equal to global max pooling.
    """
    default_hyper_params = {
        "p": 3.0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GeM, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        p = self._hyper_params["p"]

        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                fea = fea ** p
                h, w = fea.shape[2:]
                fea = fea.sum(dim=(2, 3)) * 1.0 / w / h
                fea = fea ** (1.0 / p)
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[GeM Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret

class GMP(AggregatorBase):
    """
    Global maximum pooling
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GMP, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                fea = (fea.max(dim=3)[0]).max(dim=2)[0]
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[GMP Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret

class RMAC(AggregatorBase):
    """
    Regional Maximum activation of convolutions (R-MAC).
    c.f. https://arxiv.org/pdf/1511.05879.pdf

    Hyper-Params
        level_n (int): number of levels for selecting regions.
    """

    default_hyper_params = {
        "level_n": 3,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(RMAC, self).__init__(hps)
        self.first_show = True
        self.cached_regions = dict()

    def _get_regions(self, h: int, w: int) -> List:
        """
        Divide the image into several regions.

        Args:
            h (int): height for dividing regions.
            w (int): width for dividing regions.

        Returns:
            regions (List): a list of region positions.
        """
        if (h, w) in self.cached_regions:
            return self.cached_regions[(h, w)]

        m = 1
        n_h, n_w = 1, 1
        regions = list()
        if h != w:
            min_edge = min(h, w)
            left_space = max(h, w) - min(h, w)
            iou_target = 0.4
            iou_best = 1.0
            while True:
                iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)

                # small m maybe result in non-overlap
                if iou_tmp <= 0:
                    m += 1
                    continue

                if abs(iou_tmp - iou_target) <= iou_best:
                    iou_best = abs(iou_tmp - iou_target)
                    m += 1
                else:
                    break
            if h < w:
                n_w = m
            else:
                n_h = m

        for i in range(self._hyper_params["level_n"]):
            region_width = int(2 * 1.0 / (i + 2) * min(h, w))
            step_size_h = (h - region_width) // n_h
            step_size_w = (w - region_width) // n_w

            for x in range(n_h):
                for y in range(n_w):
                    st_x = step_size_h * x
                    ed_x = st_x + region_width - 1
                    assert ed_x < h
                    st_y = step_size_w * y
                    ed_y = st_y + region_width - 1
                    assert ed_y < w
                    regions.append((st_x, st_y, ed_x, ed_y))

            n_h += 1
            n_w += 1

        self.cached_regions[(h, w)] = regions
        return regions

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                h, w = fea.shape[2:]
                final_fea = None
                regions = self._get_regions(h, w)
                for _, r in enumerate(regions):
                    st_x, st_y, ed_x, ed_y = r
                    region_fea = (fea[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]
                    region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)
                    if final_fea is None:
                        final_fea = region_fea
                    else:
                        final_fea = final_fea + region_fea
                ret[key + "_{}".format(self.__class__.__name__)] = final_fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[RMAC Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret

class SCDA(AggregatorBase):
    """
    Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval.
    c.f. http://www.weixiushen.com/publication/tip17SCDA.pdf
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(SCDA, self).__init__(hps)
        self.first_show = True

    def bfs(self, x: int, y: int, mask: torch.tensor, cc_map: torch.tensor, cc_id: int) -> int:
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        q = queue.LifoQueue()
        q.put((x, y))

        ret = 1
        cc_map[x][y] = cc_id

        while not q.empty():
            x, y = q.get()

            for (dx, dy) in dirs:
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                    if mask[new_x][new_y] == 1 and cc_map[new_x][new_y] == 0:
                        q.put((new_x, new_y))
                        ret += 1
                        cc_map[new_x][new_y] = cc_id
        return ret

    def find_max_cc(self, mask: torch.tensor) -> torch.tensor:
        """
        Find the largest connected component of the mask。

        Args:
            mask (torch.tensor): the original mask.

        Returns:
            mask (torch.tensor): the mask only containing the maximum connected component.
        """
        assert mask.ndim == 4
        assert mask.shape[1] == 1
        mask = mask[:, 0, :, :]
        for i in range(mask.shape[0]):
            m = mask[i]
            cc_map = torch.zeros(m.shape)
            cc_num = list()

            for x in range(m.shape[0]):
                for y in range(m.shape[1]):
                    if m[x][y] == 1 and cc_map[x][y] == 0:
                        cc_id = len(cc_num) + 1
                        cc_num.append(self.bfs(x, y, m, cc_map, cc_id))

            max_cc_id = cc_num.index(max(cc_num)) + 1
            m[cc_map != max_cc_id] = 0
        mask = mask[:, None, :, :]
        return mask

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                mask = fea.sum(dim=1, keepdims=True)
                thres = mask.mean(dim=(2, 3), keepdims=True)
                mask[mask <= thres] = 0
                mask[mask > thres] = 1
                mask = self.find_max_cc(mask)
                fea = fea * mask

                gap = fea.mean(dim=(2, 3))
                gmp, _ = fea.max(dim=3)
                gmp, _ = gmp.max(dim=2)

                ret[key + "_{}".format(self.__class__.__name__)] = torch.cat([gap, gmp], dim=1)
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[SCDA Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret

class SPoC(AggregatorBase):
    """
    SPoC with center prior.
    c.f. https://arxiv.org/pdf/1510.07493.pdf
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(SPoC, self).__init__(hps)
        self.first_show = True
        self.spatial_weight_cache = dict()

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                h, w = fea.shape[2:]
                if (h, w) in self.spatial_weight_cache:
                    spatial_weight = self.spatial_weight_cache[(h, w)]
                else:
                    sigma = min(h, w) / 2.0 / 3.0
                    x = torch.Tensor(range(w))
                    y = torch.Tensor(range(h))[:, None]
                    spatial_weight = torch.exp(-((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2) / 2.0 / (sigma ** 2))
                    # if torch.cuda.is_available():
                    #     spatial_weight = spatial_weight.cuda()
                    spatial_weight = spatial_weight[None, None, :, :]
                    self.spatial_weight_cache[(h, w)] = spatial_weight
                fea = (fea * spatial_weight).sum(dim=(2, 3))
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[SPoC Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret


AGGREGATORS = {
    'Crow': Crow,
    'GAP': GAP,
    'GeM': GeM,
    'GMP': GMP,
    'RMAC': RMAC,
    'SCDA': SCDA,
    'SPoC': SPoC
}

def build_aggregator(name, **kwargs):
    if name not in AGGREGATORS.keys():
        raise KeyError("Invalid aggregator, got '{}', but expected to be one of {}".format(name, AGGREGATORS.keys()))
    aggregator = AGGREGATORS[name](**kwargs)
    return aggregator

if __name__ == '__main__':
    model = build_aggregator('GeM')