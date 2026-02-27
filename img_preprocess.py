import torch.nn.functional as F
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as T


def resize_img(img, target_size):
    '''
    Resizes an image to a target size. 

    Parameters: 
        img (torch.tensor): the image tensor to be resized
        target_size (int): the target height and width for resizing. 

    Returns:
        torch.tensor: the resized image with padding. 
    '''
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
    '''
    Resizes a mask to a target size. 

    Parameters: 
        img (torch.tensor): the mask tensor to be resized
        target_size (int): the target height and width for resizing. 

    Returns:
        torch.tensor: the resized mask with padding. 
    '''
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
    '''
    Displays an image along with its ID number. 
    
    Parameters:
        img (torch.tensor): the image to be displayed. 
        number (int): the ID number of the image.
    
    Returns:
        None
    '''

    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Image {number}')
    plt.show()


def to_tensor(img):
    '''
    Converts and image to a torch.tensor. 

    Parameters:
        img (numpy.array): image to be converted.
    
    Returns:
        torch.tensor: the image as a tensor. 
    '''

    transform = T.Compose([T.ToTensor(),])
    return transform(img)
