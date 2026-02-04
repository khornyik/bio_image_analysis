import torch.nn.functional as F
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as T


def resize_img(img, target_size):
    print(img.shape)
    _, height, width = img.shape
    target_height = target_size
    target_width = target_size

    scale = min(target_height / height, target_width / width)
    new_height, new_width = int(scale * height), int(scale * width)

    print(new_height, new_width)

    img = TF.resize(img, size = (new_height, new_width))

    h_padding = int(target_height - new_height)
    w_padding = int(target_width - new_width)

    padding = (
        w_padding // 2, h_padding // 2, w_padding - w_padding // 2, h_padding - h_padding // 2
    )

    return F.pad(img, padding, value = 0)



def resize_mask(img, target_size):
    print(img.shape)
    _, height, width = img.shape
    target_height = target_size
    target_width = target_size

    scale = min(target_height / height, target_width / width)
    new_height, new_width = int(scale * height), int(scale * width)

    print(new_height, new_width)

    img = TF.resize(img, size = (new_height, new_width), interpolation = TF.InterpolationMode.NEAREST)

    h_padding = int(target_height - new_height)
    w_padding = int(target_width - new_width)

    padding = (
        w_padding // 2, h_padding // 2, w_padding - w_padding // 2, h_padding - h_padding // 2
    )

    return F.pad(img, padding, value = 0)



def display_img(img, number):

    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Image {number}')
    plt.show()


def to_tensor(img):

    transform = T.Compose([T.ToTensor(),])
    return transform(img)
