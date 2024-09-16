import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torchvision.transforms.v2 import Normalize

def show(imgs, zoom):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)

def plot_grid(images, nrow=8, zoom=2):

    grid = make_grid(images, nrow=nrow)
    show(grid, zoom=zoom)

imagenet_normalize = Normalize(IMAGENET_NORMALIZE['mean'], IMAGENET_NORMALIZE['std'])