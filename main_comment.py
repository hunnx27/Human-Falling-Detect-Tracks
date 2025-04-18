import os
import cv2
import time
import torch
import argparse
import numpy as np

# 필요한 모듈 및 클래스 임포트
from Detection.Utils import ResizePadding  # 이미지 크기 조정 및 패딩 유틸리티
from CameraLoader import CamLoader, CamLoader_Q  # 카메라 또는 비디오 로더 클래스
from DetectorLoader import TinyYOLOv3_onecls  # 사람 감지를 위한 YOLO 모델 로더

from PoseEstimateLoader import SPPE_FastPose  # 자세 추정 모델 로더
from fn import draw_single  # 골격 그리기 함수

from Track.Tracker import Detection, Tracker  # 객체 추적 관련 클래스
from ActionsEstLoader import TSSTG  # 행동 인식 모델 로더

# 기본 비디오 소스 경로 설정
source = '../Data/falldata/Home/Videos/video (1).avi'

def preproc(image):
    """
    CameraLoader를 위한 전처리 함수.
    입력 이미지의 크기를 조정하고 색상 형식을 변환합니다.
    """
    image = resize_fn(image)  # 이미지 크기 조정
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 색상 형식 변환
    return image


def kpt2bbox(kpt, ex=20):
    """
    모든 키포인트(x,y)를 포함하는 경계 상자(bounding box) 생성
    kpt: `(N, 2)` 형태의 배열, 키포인트 좌표
    ex: (int) 경계 상자 확장 크기
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    # 명령줄 인자 파서 설정
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='카메라 소스 또는 비디오 파일 경로.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='감지 모델의 입력 크기 (정사각형, 32로 나눌 수 있어야 함)')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='자세 모델의 입력 크기 (32로 나눌 수 있어야 함, 높이x너비)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='SPPE FastPose 모델의 백본 모델')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='모든 감지된 경계 상자 표시 여부')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='골격 자세 표시 여부')
    par.add_argument('--save_out', type=str, default='',
                        help='처리된 영상을 비디오 파일로 저장')
    par.add_argument('--device', type=str, default='cuda',
                        help='모델 실행 디바이스 (cpu 또는 cuda)')
    args = par.parse_args()  # 명령줄 인자 파싱

    device = args.device  # 실행 디바이스 설정

    # 사람 감지 모델 초기화
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # 자세 추정 모델 초기화
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # 추적기(Tracker) 초기화
    max_age = 30  # 최대 추적 기간 (프레임 수)
    tracker = Tracker(max_age=max_age, n_init=3)  # 3프레임 이상 감지되어야 추적 시작

    # 행동 인식 모델 초기화
    action_model = TSSTG()

    # 이미지 크기 조정 함수 초기화
    resize_fn = ResizePadding(inp_dets, inp_dets)

    # 카메라 또는 비디오 소스 설정
    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # 비디오 파일인 경우 큐를 사용하는 로더 스레드 사용
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # 웹캠인 경우 일반 스레드 로더 사용
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # 비디오 출력 설정
    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')  # 비디오 코덱 설정
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))  # 비디오 작성기 초기화

    fps_time = 0  # FPS 계산용 시간 변수
    f = 0  # 프레임 카운터
    
    # 메인 처리 루프
    while cam.grabbed():  # 카메라에서 프레임을 성공적으로 가져오는 동안 반복
        f += 1  # 프레임 카운터 증가
        frame = cam.getitem()  # 현재 프레임 가져오기
        image = frame.copy()  # 원본 이미지 복사

        # 감지 모델을 사용하여 프레임에서 사람 감지
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Kalman 필터를 사용하여 이전 프레임 정보로부터 현재 프레임의 위치 예측
        tracker.predict()
        
        # 이전 추적 정보를 현재 감지 결과와 병합
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # 추적을 위한 Detection 객체 리스트
        if detected is not None:
            # 각 경계 상자에 대한 골격 자세 예측
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Detection 객체 생성 (키포인트, 점수 등 포함)
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # 감지된 경계 상자 시각화 (옵션)
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # 현재 프레임과 이전 프레임의 정보를 매칭하여 추적 업데이트
        tracker.update(detections)

        # 각 추적 대상의 행동 예측
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():  # 확정된 추적만 처리
                continue

            track_id = track.track_id  # 추적 ID
            bbox = track.to_tlbr().astype(int)  # 경계 상자 좌표
            center = track.get_center().astype(int)  # 중심점 좌표

            action = 'pending..'  # 기본 행동 텍스트
            clr = (0, 255, 0)  # 기본 색상 (녹색)
            
            # 30프레임 시간 단계로 행동 예측
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])  # 행동 예측
                action_name = action_model.class_names[out[0].argmax()]  # 가장 확률 높은 행동 이름
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)  # 행동 및 확률 텍스트
                
                # 행동에 따른 색상 변경
                if action_name == 'Fall Down':  # 낙상인 경우
                    clr = (255, 0, 0)  # 빨간색
                elif action_name == 'Lying Down':  # 누워있는 경우
                    clr = (255, 200, 0)  # 주황색

            # 추적 결과 시각화
            if track.time_since_update == 0:  # 현재 프레임에서 업데이트된 추적만 표시
                if args.show_skeleton:  # 골격 표시 옵션이 켜져 있으면
                    frame = draw_single(frame, track.keypoints_list[-1])  # 골격 그리기
                    
                # 경계 상자 그리기
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                # 추적 ID 표시
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                # 행동 분류 결과 표시
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # 프레임 표시 준비
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)  # 표시용으로 프레임 크기 2배 확대
        # FPS 정보 표시
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]  # 색상 채널 순서 변환 (BGR -> RGB)
        fps_time = time.time()  # FPS 계산을 위한 시간 업데이트

        # 비디오 저장 옵션이 켜져 있으면 프레임 저장
        if outvid:
            writer.write(frame)

        # 화면에 프레임 표시
        cv2.imshow('frame', frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 정리
    cam.stop()  # 카메라/비디오 스트림 정지
    if outvid:
        writer.release()  # 비디오 작성기 해제
    cv2.destroyAllWindows()  # 모든 창 닫기