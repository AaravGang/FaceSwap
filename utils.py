import requests
import numpy as np
from io import BytesIO
from PIL import Image


def url_to_img(url):
    img = Image.open(BytesIO(requests.get(url).content))
    return np.array(img)
