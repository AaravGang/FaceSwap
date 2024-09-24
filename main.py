# from bson import json_util
from flask import Flask, request
from utils import url_to_img, url_to_video
import cv2
import matplotlib.pyplot as plt
import mediapipe  # mediapipe doesnt work on old af mac
import itertools
import numpy as np
from constants import *
from connect import upload_img, upload_video
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import os
from proglog import ProgressBarLogger


class MyBarLogger(ProgressBarLogger):

    def __init__(self, progress, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress

    def callback(self, **changes):
        # Every time the logger message is updated, this function is called with
        # the `changes` dictionary of the form `parameter: new value`.
        for (parameter, value) in changes.items():
            self.progress['Writing Video']['name'] = value[9:]

    def bars_callback(self, bar, attr, value, old_value=None):
        # Every time the logger progress is updated, this function is called
        percentage = int((value / self.bars[bar]['total']) * 100)
        self.progress['Writing Video'][
            'value'] = f"|{'█'*(percentage//5)+'　'*(20-(percentage//5))}| {percentage}%"


# load images
# src_orig = cv2.imread("./src.png")
# dst_orig = cv2.imread("./dst.png")

# load face mesh module
faceModule = mediapipe.solutions.face_mesh
face_mesh = faceModule.FaceMesh(static_image_mode=True, max_num_faces=5)

progress = {}
working = False


# get landmarks
def get_landmarks(img):
    results = face_mesh.process(img)
    landmarks = results.multi_face_landmarks
    return landmarks


# triangulate the face
def triangulate(landmarks, width, height):
    for idx_triangle in INDICES:
        triangle = np.array([(int(width * landmarks.landmark[idx].x),
                              int(height * landmarks.landmark[idx].y))
                             for idx in idx_triangle], np.int32)

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
        triangulate(dst_landmarks, dst.shape[1], dst.shape[0])
    ]

    for _ in range(NUM_TRIANGLES):
        src_triangle = next(triangulations[0])
        dst_triangle = next(triangulations[1])

        (x, y, src_w, src_h) = cv2.boundingRect(src_triangle)
        src_cropped = src[y:y + src_h, x:x + src_w]
        src_normalised = np.float32(src_triangle - [x, y])

        dst_rect = [x, y, w, h] = cv2.boundingRect(dst_triangle)

        dst_normalised = np.int32(dst_triangle - [x, y])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, dst_normalised, 255)

        dst_normalised = np.float32(dst_normalised)

        M = cv2.getAffineTransform(src_normalised, dst_normalised)
        warped_triangle = cv2.warpAffine(src_cropped, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle,
                                          warped_triangle,
                                          mask=mask)

        yield warped_triangle, dst_rect


def get_landmark_points(landmarks, width, height):
    return np.array(
        [[(width * landmarks.landmark[idx].x, height * landmarks.landmark[idx].y)]
         for idx in range(len(landmarks.landmark))], np.int32)


def face_swap(src, dst, src_landmarks, dst_landmarks):
    face = np.zeros_like(dst)

    for warped_triangle, dst_rect in transform(src, dst, src_landmarks,
                                               dst_landmarks):
        x, y, w, h = dst_rect
        sub_face = face[y:y + h, x:x + w]
        if sub_face.shape[0] == 0:
            continue

        sub_face_gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)

        _, mask_triangles_designed = cv2.threshold(sub_face_gray, 1, 255,
                                                   cv2.THRESH_BINARY_INV)

        warped_triangle = cv2.bitwise_and(warped_triangle,
                                          warped_triangle,
                                          mask=mask_triangles_designed)

        sub_face = cv2.add(sub_face, warped_triangle)
        face[y:y + h, x:x + w] = sub_face

    dst_mask = np.zeros_like(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY))
    dst_face_convexHull = cv2.convexHull(
        get_landmark_points(dst_landmarks, dst.shape[1], dst.shape[0]))
    dst_face_mask = cv2.fillConvexPoly(dst_mask, dst_face_convexHull, 255)

    dst_body_mask = cv2.bitwise_not(dst_face_mask)
    dst_body = cv2.bitwise_and(dst, dst, mask=dst_body_mask)

    result = cv2.add(face, dst_body)
    (x, y, w, h) = cv2.boundingRect(dst_face_convexHull)
    dst_face_center = (int((2 * x + w) / 2), int((2 * y + h) / 2))

    result = cv2.seamlessClone(result, dst, dst_face_mask, dst_face_center,
                               cv2.NORMAL_CLONE)

    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result


def main(src, dst, show=False, upload=True):

    # dst_ = dst.copy()

    src_multi_landmarks = get_landmarks(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    dst_multi_landmarks = get_landmarks(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

    if all([src_multi_landmarks, dst_multi_landmarks]):

        for i in range(len(dst_multi_landmarks)):

            src_landmarks = src_multi_landmarks[0]
            dst_landmarks = dst_multi_landmarks[i]

            dst = face_swap(src, dst, src_landmarks, dst_landmarks)

            # dst_ = plot_landmark(dst_, dst_landmarks)

    else:
        # print("No Faces Detected", type(src_multi_landmarks),
        # type(dst_multi_landmarks))
        return False

    if show:
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

    if upload:
        upload_id = upload_img(dst)
        return upload_id

    return dst


def handle_video(src, cap, request_id, logger):
    res_path = 'temp.mp4'

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_count = 0

    result = cv2.VideoWriter(res_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

    for c in range(count):
        ret, frame = cap.read()
        if ret:
            try:
                res = main(src, frame, upload=False)
            except:
                result.release()
            result.write(frame if type(res) == bool and res == False else res)
            if type(res) != bool:
                face_count += 1

        else:
            break

        progress[request_id]['Editing'][
            'value'] = f"|{'█'*int(c/count * 20)+'－'*(20-int(c/count * 20))}| {int(c/count * 100)}%"

    cap.release()
    result.release()

    if face_count == 0:
        return False

    progress[request_id]['Editing']['value'] = f"|{'█'*20}| 100%"
    audio_clip = get_audio("input.mp4")
    video_clip = VideoFileClip("temp.mp4")
    video_clip = video_clip.set_audio(audio_clip)
    progress[request_id]['Loading Audio']['value'] = "Done"
    progress[request_id]['Writing Video']['value'] = "Processing."
    video_clip.write_videofile("result.mp4", logger=logger)
    # progress[request_id]['Writing Video']['value'] = "Done"

    return upload_video(res_path)


def get_audio(filename):
    clip = VideoFileClip(filename)
    return clip.audio


app = Flask(__name__)


@app.route("/result", methods=["GET", "POST"])
def result():
    return {"Status": "Working Well"}


@app.route('/')  # '/' for the default page
def home():
    return "Wow this is a basic output!"


@app.route("/api", methods=["GET", "POST"])
def api():
    src_url = request.args.get("srcUrl")
    dst_url = request.args.get("dstUrl")

    # print(src_url, dst_url)
    dst = url_to_img(dst_url)
    src = url_to_img(src_url)

    if (type(dst) == bool and dst == False) or (type(src) == bool
                                                and src == False):
        return {'status': False}

    dst = cv2.cvtColor(dst, cv2.COLOR_RGBA2BGR)
    src = cv2.cvtColor(src, cv2.COLOR_RGBA2BGR)

    id = main(src, dst)
    return {'status': bool(id), 'id': str(id)}


@app.route("/video", methods=["GET", "POST"])
def video_api():
    global working
    if working:
        return {
            'status': False,
            'error': 'Try again later. Working on another video...'
        }

    working = True
    src_url = request.args.get("srcUrl")
    dst_url = request.args.get("dstUrl")
    request_id = request.args.get("request_id")

    print(src_url, dst_url)

    src = cv2.cvtColor(url_to_img(src_url), cv2.COLOR_RGBA2BGR)
    cap = url_to_video(dst_url)
    if not cap:
        return

    progress[request_id] = {
        'Editing': {
            'name': 'Editing',
            'value': f'|{"—"*20}| 0%',
            'inline': False,
        },
        'Loading Audio': {
            'name': 'Loading Audio',
            'value': '0%',
            'inline': False,
        },
        'Writing Video': {
            'name': 'Writing Video',
            'value': '0%',
            'inline': False,
        },
        'Uploading': {
            'name': 'Uploading',
            'value': '0%',
            'inline': False,
        }
    }

    logger = logger = MyBarLogger(progress[request_id])

    id = handle_video(src, cap, request_id, logger)

    progress[request_id]['Uploading']['value'] = 'Done'

    if id:
        os.remove("input.mp4")
        os.remove("temp.mp4")
        os.remove("result.mp4")

    working = False

    return {'status': bool(id), 'id': str(id)}


@app.route('/progress/<int:id>', methods=['GET', 'POST'])
def progress_api(id):
    return progress.get(str(id)) if progress.get(str(id)) else {
        'progress': {
            'name': 'Progress',
            'value': 'Not Started',
            'inline': True,
        }
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
