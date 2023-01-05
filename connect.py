import os
from pymongo import MongoClient
from PIL import Image
import io
import cv2
import base64, json

MONGODB_SRV = os.getenv("MONGODB_SRV")

# Create a connection using MongoClient.
client = MongoClient(MONGODB_SRV)
# the database
db = client.details
# the collection
img_storage = db["image-storage"]


def upload_img(img):
  img = cv2.resize(img,(500,500))
  ret, buf = cv2.imencode('.png', img)
  image = base64.b64encode(buf).decode('ascii')
  print(type(image))
  data = {'buffer': image}
  image_id = img_storage.insert_one(data).inserted_id
  print("uploaded")
  return image_id
