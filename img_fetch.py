from functools import lru_cache

def make_image_fetch(session):

    @lru_cache(maxsize = 130)
    def get_image_bytes(image_id: str):

        base_url = 'https://idr.openmicroscopy.org'

        render_image_url = f'{base_url}/webgateway/render_image/{image_id}/0/0/?m=c&c=0|100:250$00FF00,1|100:250$FF0000,2|100:250$9ACD32,3|100:250$0000FF'
        qs = {'base': base_url, 'image_id': image_id}
        image_url = render_image_url.format(**qs)
        r_img = session.get(image_url)
        if r_img.status_code != 200:
            r_img.raise_for_status()

        render_mask_url = f'{base_url}/webgateway/render_image/{image_id}/0/0/?m=c&c=0|100:250$000000,1|100:250$000000,2|100:250$000000,3|100:250$FFFFFF'
        mask_url = render_mask_url.format(**qs)
        r_mask = session.get(mask_url)
        if r_mask.status_code != 200:
            r_mask.raise_for_status()


        return r_img.content, r_mask.content
    return get_image_bytes
