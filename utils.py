import requests
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile
import cv2


def url_to_img(url):
    img = Image.open(BytesIO(requests.get(url).content))
    return np.array(img)


def url_to_video(url):
    data = requests.get(url).content

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(data)
        cap = cv2.VideoCapture(temp.name)
        temp.close()
    return cap

