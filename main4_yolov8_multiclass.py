# requirements:
# pip install ultralytics opencv-python torch numpy

import cv2
import numpy as np
from ultralytics import YOLO
from PoseEstimateLoader import SPPE_FastPose  # 기존 사용하던 포즈 추정 모듈
from ActionsEstLoader import TSSTG            # 행동 인식 모델

# 1. YOLOv8 모델 로드 (multi-class 모델, 예: doctor=0, patient=1)
model = YOLO("best.pt")  # 학습된 다중 클래스 모델

# 2. Pose 및 Action 모델 초기화
pose_model = SPPE_FastPose("resnet50", 224, 160, device="cuda")
action_model = TSSTG()

def kpt2bbox(kpt, ex=20):
    return np.array([kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex])

# 3. 영상 처리 루프
cap = cv2.VideoCapture("your_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLOv8 추론
    results = model(frame)[0]

    # 5. patient만 필터링
    patient_boxes = []
    patient_scores = []
    for box in results.boxes:
        cls = int(box.cls)
        if model.names[cls] == "patient":  # 또는 cls == 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf.cpu().numpy())
            patient_boxes.append([x1, y1, x2, y2])
            patient_scores.append(score)

    if not patient_boxes:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        continue

    # 6. Pose 추정
    boxes = np.array(patient_boxes)
    scores = np.array(patient_scores)
    keypoints = pose_model.predict(frame, boxes, scores)

    for kp in keypoints:
        # 7. 행동 예측
        pts = np.array([kp["keypoints"].numpy()], dtype=np.float32)
        action_result = action_model.predict(pts, frame.shape[:2])
        action_name = action_model.class_names[action_result[0].argmax()]
        confidence = action_result[0].max()

        # 8. 시각화
        bbox = kpt2bbox(kp["keypoints"].numpy())
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{action_name}: {confidence:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()