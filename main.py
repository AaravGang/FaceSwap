from flask import Flask, request
from utils import url_to_img
import cv2
import matplotlib.pyplot as plt
import mediapipe
import itertools
import numpy as np
from constants import *
from connect import upload_img


# load images
# src_orig = cv2.imread("./src.png")
# dst_orig = cv2.imread("./person.jpg")

# load face mesh module
faceModule = mediapipe.solutions.face_mesh
face_mesh = faceModule.FaceMesh(static_image_mode=True, max_num_faces=5)


# get landmarks
def get_landmarks(img):
    results = face_mesh.process(img)
    landmarks = results.multi_face_landmarks
    return landmarks


# triangulate the face
def triangulate(landmarks, width, height):
    for idx_triangle in INDICES:
        triangle = np.array(
            [
                (
                    int(width * landmarks.landmark[idx].x),
                    int(height * landmarks.landmark[idx].y),
                )
                for idx in idx_triangle
            ],
            np.int32,
        )

        yield triangle


# show tesselated face
def plot_landmark(img_base, landmarks):

    img = img_base.copy()
    width, height = img.shape[1], img.shape[0]
    color = (255, 255, 255)

    for triangle in triangulate(landmarks, width, height):
        for sub in itertools.combinations(triangle, 2):
            cv2.line(img, sub[0], sub[1], color, thickness=2)

    return img


def transform(src, dst, src_landmarks, dst_landmarks):
    triangulations = [
        triangulate(src_landmarks, src.shape[1], src.shape[0]),
        triangulate(dst_landmarks, dst.shape[1], dst.shape[0]),
    ]

    for _ in range(NUM_TRIANGLES):
        src_triangle = next(triangulations[0])
        dst_triangle = next(triangulations[1])

        (x, y, src_w, src_h) = cv2.boundingRect(src_triangle)
        src_cropped = src[y : y + src_h, x : x + src_w]
        src_normalised = np.float32(src_triangle - [x, y])

        dst_rect = [x, y, w, h] = cv2.boundingRect(dst_triangle)

        dst_normalised = np.int32(dst_triangle - [x, y])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, dst_normalised, 255)

        dst_normalised = np.float32(dst_normalised)

        M = cv2.getAffineTransform(src_normalised, dst_normalised)
        warped_triangle = cv2.warpAffine(src_cropped, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask)

        yield warped_triangle, dst_rect


def get_landmark_points(landmarks, width, height):
    return np.array(
        [
            [(width * landmarks.landmark[idx].x, height * landmarks.landmark[idx].y)]
            for idx in range(len(landmarks.landmark))
        ],
        np.int32,
    )


def face_swap(src, dst, src_landmarks, dst_landmarks):
    face = np.zeros_like(dst)

    for warped_triangle, dst_rect in transform(src, dst, src_landmarks, dst_landmarks):
        x, y, w, h = dst_rect
        sub_face = face[y : y + h, x : x + w]
        if sub_face.shape[0] == 0:
            continue

        sub_face_gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)

        _, mask_triangles_designed = cv2.threshold(
            sub_face_gray, 1, 255, cv2.THRESH_BINARY_INV
        )

        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=mask_triangles_designed
        )

        sub_face = cv2.add(sub_face, warped_triangle)
        face[y : y + h, x : x + w] = sub_face

    dst_mask = np.zeros_like(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY))
    dst_face_convexHull = cv2.convexHull(
        get_landmark_points(dst_landmarks, dst.shape[1], dst.shape[0])
    )
    dst_face_mask = cv2.fillConvexPoly(dst_mask, dst_face_convexHull, 255)

    dst_body_mask = cv2.bitwise_not(dst_face_mask)
    dst_body = cv2.bitwise_and(dst, dst, mask=dst_body_mask)

    result = cv2.add(face, dst_body)
    (x, y, w, h) = cv2.boundingRect(dst_face_convexHull)
    dst_face_center = (int((2 * x + w) / 2), int((2 * y + h) / 2))

    result = cv2.seamlessClone(
        result, dst, dst_face_mask, dst_face_center, cv2.NORMAL_CLONE
    )

    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result


def main(src, dst, show=False):

    # dst_ = dst.copy()

    src_multi_landmarks = get_landmarks(src)

    dst_multi_landmarks = get_landmarks(dst)

    if all([src_multi_landmarks, dst_multi_landmarks]):

        for i in range(len(dst_multi_landmarks)):

            src_landmarks = src_multi_landmarks[0]
            dst_landmarks = dst_multi_landmarks[i]

            dst = face_swap(src, dst, src_landmarks, dst_landmarks)

            # dst_ = plot_landmark(dst_, dst_landmarks)

    else:
        print("No Faces Detected", type(src_multi_landmarks), type(dst_multi_landmarks))
        return False

    if show:
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

    upload_id = upload_img(dst)

    return upload_id


app = Flask(__name__)


@app.route("/result", methods=["GET", "POST"])
def result():
    return {"Status": "Working Well"}


@app.route("/")  # '/' for the default page
def home():
    return "Wow this is a basic output!"


@app.route("/api", methods=["GET", "POST"])
def api():
    dst_url = request.args.get("dstUrl")
    src_url = request.args.get("srcUrl")

    # print(src_url, dst_url)

    dst = cv2.cvtColor(url_to_img(dst_url), cv2.COLOR_RGBA2BGR)
    src = cv2.cvtColor(url_to_img(src_url), cv2.COLOR_RGBA2BGR)

    res = main(src, dst)
    print(res)
    return {"status": bool(res), "id": str(res)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

