import os
import os.path as osp

from torch.utils.data import Dataset
from PIL import Image

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset_dir='/home/bk/general_sys/data/gallery', transform=None):
        self.dataset_dir = dataset_dir
        self.img_paths = list(map(lambda x : osp.join(self.dataset_dir, x), os.listdir(self.dataset_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = read_image(self.img_paths[index])
        img_id = self.img_paths[index].split('/')[-1]
        if self.transform is not None:
            img = self.transform(img)
        return img, img_id