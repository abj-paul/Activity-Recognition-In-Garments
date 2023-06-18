import cv2
import os
from ultralytics import YOLO
import supervision as sv
import numpy as np


LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)

def create_folders(detections, model):
    class_id_list = [ model.names[detections.class_id[i]]+str(detections.tracker_id[i]) for i in range(len(detections)) ]

    if not os.path.exists('data/'):
        os.mkdir('data/')
    for class_id in class_id_list:
        class_folder = f'data/registered_entity_{class_id}'
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)
    




def main():
    # line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    # line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    model = YOLO("yolov8n.pt")
    model.fuse()
    for frame_num, result in enumerate(model.track(
        source="video.mp4", show=False, stream=True, agnostic_nms=True, persist=True
    )):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        labels = [
            f"{tracker_id} {model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        # line_counter.trigger(detections=detections)
        # line_annotator.annotate(frame=frame, line_counter=line_counter)

        create_folders(detections, model)

        count = 0
        for i in range(len(detections)):
            print(detections[i])
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            crop = frame[y1:y2, x1:x2]

            class_folder = f'data/registered_entity_{model.names[detections.class_id[i]]+str(detections.tracker_id[i])}'
            cv2.imwrite(f"{class_folder}/bbox{frame_num}_{count}.jpg", crop)
            count += 1

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()