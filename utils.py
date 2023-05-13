import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from io import BytesIO
from PIL import Image
import cv2


def url_to_img(url):
    try:
        img = requests.get(url)
        img = BytesIO(img.content)
        img = Image.open(img)
    except Exception as e:
        print("Exception in url to image", e)
        return False
    return np.array(img)


def url_to_video(url):
    try:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        data = session.get(url).content

        with open("input.mp4", "wb") as f:
            f.write(data)
            cap = cv2.VideoCapture(f.name)
            f.close()
        return cap
    except Exception as e:
        print("Exception in url to video", e)
        return False
