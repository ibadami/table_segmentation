import json
import os.path

import numpy as np
from cv2 import cv2


def draw_table(img, points):
    if len(points) > 0:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [points], -1, (0, 255, 255), 3)
        for pt in points:
            img = cv2.circle(img, (pt[0][0], pt[0][1]), 3, (0, 0, 255), -1)
    return img


def resize_img(img, scale_factor=4):
    im_height = int(img.shape[0] / scale_factor)
    im_width = int(img.shape[1] / scale_factor)
    dim = (im_width, im_height)
    im_down = cv2.resize(img, dim).astype(np.uint8)
    return im_down

def create_video_from_images(img_dir):
    pass


def create_video_from_detection(detections, video_path):
    pass


def create_video(image_dir, table_segments_file_path, video_path):

    with open(table_segments_file_path, 'r') as file:
        detections = json.load(file)
    video_out_path = os.path.join(image_dir, image_dir.split('/')[-1:]+'.mp4')
    os.makedirs(video_out_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    scale_factor = 2.0
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/scale_factor)
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/scale_factor)
    dim = (im_width, im_height)
    frame_no = 0
    while 1:
        ret, frame = cap.read()
        if ret:
            frame_no += 1
            mask = cv2.imread(mask_file_path)
            points = np.array(detections[str(frame_no)])
            img = draw_table(frame, points)
            im_down = cv2.resize(img, dim).astype(np.uint8)
            mask_down = cv2.resize(mask, dim).astype(np.uint8)
            detection_frame = cv2.hconcat([im_down, mask_down])
            cv2.imwrite(file_path, detection_frame)
        else:
            break


if __name__ == "__main__":

    video_path = 'dataset/videos/LuckyLadies.mp4'
    image_output_path = './result/masks'
    detection_file_path = './result/detections.json'
    create_video(video_path, mask_output_path, detection_file_path)

