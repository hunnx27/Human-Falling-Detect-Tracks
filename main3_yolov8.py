import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from Detection.Utils import ResizePadding

# 모델 로딩
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("path/to/yolov8n.pt").to(device)  # 학습된 모델 경로
pose_model = SPPE_FastPose('resnet50', 224, 160, device=device)
action_model = TSSTG()
tracker = Tracker(max_age=30, n_init=3)

resize_fn = ResizePadding(640, 640)

def kpt2bbox(kpt, ex=20):
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

# 영상 입력
cap = cv2.VideoCapture('your_video.mp4')
fps_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = resize_fn(frame.copy())
    results = yolo_model(img, verbose=False)[0]

    detections = []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0 or conf < 0.3:
            continue

        bbox = [int(x1), int(y1), int(x2), int(y2)]
        pose = pose_model.predict(frame, torch.tensor([bbox]), torch.tensor([conf]))
        for ps in pose:
            kpts = ps['keypoints'].numpy()
            scores = ps['kp_score'].numpy()
            detections.append(Detection(
                kpt2bbox(kpts),
                np.concatenate((kpts, scores), axis=1),
                scores.mean()
            ))

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr().astype(int)
        center = track.get_center().astype(int)
        action = "pending.."
        color = (0, 255, 0)

        if len(track.keypoints_list) == 30:
            pts = np.array(track.keypoints_list, dtype=np.float32)
            out = action_model.predict(pts, frame.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            action = f'{action_name}: {out[0].max() * 100:.2f}%'
            if action_name == 'Fall Down':
                color = (255, 0, 0)
            elif action_name == 'Lying Down':
                color = (255, 200, 0)

        frame = draw_single(frame, track.keypoints_list[-1])
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"FPS: {1.0 / (time.time() - fps_time):.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    fps_time = time.time()

    cv2.imshow("YOLOv8 Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()