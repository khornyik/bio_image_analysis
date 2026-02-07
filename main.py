# python -m pip install requests
# python -m pip install functools 

# python -m pip install torch torchvision - includes sympy, setuptools, pillow, numpy, networkx, 
# MarkupSafe, fsspec, filelock, jinja2, torch, torchvision

# python -m pip install matplotlib

import requests
from io import BytesIO
from PIL import Image
from skimage import exposure
from skimage.filters import gaussian
from img_preprocess import resize_img, resize_mask, display_img, to_tensor
from img_fetch import make_image_fetch


index_page = '%s/webclient/?experimenter=-1' % 'https://idr.openmicroscopy.org'


def img_to_tensor(img_bytes):

    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    return to_tensor(img)


def main():

    image_ids = []

    dataset_id = 18801

    with requests.Session() as session:

        request = requests.Request('GET', index_page)
        prepped = session.prepare_request(request)
        response = session.send(prepped)
        if response.status_code != 200:
            response.raise_for_status()


        for i in range(0, 5):
            dataset_url = f'https://idr.openmicroscopy.org/api/v0/m/datasets/{dataset_id}/images/' 
            resp = requests.get(dataset_url)
            ds_img_ids = [i['@id'] for i in resp.json()['data']]
            for i in ds_img_ids:
                image_ids.append(i)

            dataset_id += 1

        get_image_bytes = make_image_fetch(session)


        print('length of image ids:', len(image_ids))
        print(image_ids)

        
        for i, image_id in enumerate(image_ids, start = 1):
            img_bytes, mask_bytes = get_image_bytes(image_id)

            img_tensor = img_to_tensor(img_bytes)
            mask_tensor = img_to_tensor(mask_bytes)

            img_tensor = resize_img(img_tensor, 512)   # 512, 256, 128, 64
            mask_tensor = resize_mask(mask_tensor, 512)    # 512, 256, 128, 64

            if i == 5:

                display_img(img_tensor.permute(1, 2, 0), image_id)

                img_tensor_filterd = gaussian(img_tensor.permute(1, 2, 0).numpy(), sigma = 0.7)
                img_tensor_enhanced = to_tensor(exposure.equalize_adapthist(img_tensor_filterd, clip_limit = 0.03)).permute(1, 2, 0)

                display_img(img_tensor_enhanced, image_id)

                print(img_tensor_enhanced.shape)

                mask_tensor_filtered = gaussian(mask_tensor.permute(1, 2, 0).numpy(), sigma = 0.7)
                mask_tensor_enahnced = to_tensor(exposure.equalize_adapthist(mask_tensor_filtered, clip_limit = 0.01)).permute(1, 2, 0)

                display_img(mask_tensor_enahnced, image_id)
                break


if __name__ == '__main__':
    main()