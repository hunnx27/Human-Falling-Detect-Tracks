# 낙상 사고 방지 왜 필요한가?
1. 노인 낙상 사고 사망 비율 65세 이상 33%, 80세 이상 50%
2. 노인인구 증가
3. 의료 인력 부족

# 기존 현황
구호조치시간 120초 > 10초??

# 낙상방지 링크
거리기반 IOT 낙상방지 시스템 : https://at3d.or.kr/contest/view?idx=620

# 낙상 사고 위험이 있는 대상자란?
거동이 불편
야간에 혼자 움직이는 빈도가 높은 경우
야간 배뇨 습관이 있는 경우

# 다양한 낙상 감지 방식
센서를 이용한 낙상 감지 : https://blog.naver.com/tomamedi/222647371948
거리감지를 이용한 낙상감지 : https://at3d.or.kr/contest/view?idx=620
kinect를 이용한 낙상감지 : https://patents.google.com/patent/KR101438002B1/ko
바닥센서를 이용한 낙상 감지 : https://cbiz.chosun.com/svc/bulletin/bulletin_art.html?contid=2017022203183
어안CCTV를 이용한 낙상 감지(AI-PAM) : https://www.youtube.com/watch?v=dAb31ZCl8-0
라즈베리파이카메라를 이용한 낙상 감지 : https://capstone.uos.ac.kr/mie/index.php/2%EC%A1%B0-%EC%A1%B8%EC%97%85%EC%8B%9C%EC%BC%9C%EC%A1%B0#.EB.82.99.EC.83.81.EA.B0.90.EC.A7.80_.EB.B0.A9.EC.8B.9D

# 프로젝트 체택 ?
- 어안CCTV를 이용한 낙상 감지(AI-PAM) : https://www.youtube.com/watch?v=dAb31ZCl8-0
- AI기반 병동 환자 행동분석 모니터링 시스템(AI-PAM) : https://www.youtube.com/watch?v=B2B9x_F4-h4
- 라즈베리파이카메라를 이용한 낙상 감지 : https://capstone.uos.ac.kr/mie/index.php/2%EC%A1%B0-%EC%A1%B8%EC%97%85%EC%8B%9C%EC%BC%9C%EC%A1%B0#.EB.82.99.EC.83.81.EA.B0.90.EC.A7.80_.EB.B0.A9.EC.8B.9D

# 필요 준비물
- 어안 CCTV?
- 인공지능 서버?
- raspberry camera v2 8mp(1080p, 1920*1080)/30fps
- logitech c922(1080p, 1920*1080)/30fps
- 열화상카메라 : FLIR One Pro FLIR A615 AX8 욜로 연동


# 낙상 감지 알고리즘
- 선 안전요소 감지
  > 침대 이탈 했는지?
  > 난간이 내려갔는지?
  > 
- 환자 식별 : 커튼 영역과 비영역 분리(커튼 영역이 환자의 행동 범위)
- 환자 식별 : 환자복
- 낙상판단 알고리즘 : Tiny-Yolo onecla(사람감지) > AlphaPose(객체의 골격, 스켈레톤 포즈 얻기) > ST-GCN모델(액션 인식- 낙상판별)


# 부가기능
- 
- 의료진 마스킹  처리

# 모니터링
- CCTV 모니터로 알림
- 스마트폰으로 알림
- 스마트워치로 알림

# 욜로 웹 서버 연동
- https://wikidocs.net/215172
# https://made-by-kyu.tistory.com/entry/OpenCV-YOLOv8-%EC%BB%A4%EC%8A%A4%ED%85%80-%ED%95%99%EC%8A%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%A7%8C%EB%93%A4%EA%B8%B01
yolo8 커스텀 학습하기

# 레이블링
1. 사람 식별하기, 침대 식별하기, 커튼 식별하기
2. 의사와 사람 식별하기(커튼밖과 안, 의복)

# yolo8 > 알파포즈 > st-gcn
https://github.com/GajuuzZ/Human-Falling-Detect-Tracks
> https://drive.google.com/drive/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO
  >> https://jjuke-brain.tistory.com/entry/GPU-%EC%84%9C%EB%B2%84-%EC%82%AC%EC%9A%A9%EB%B2%95-CUDA-PyTorch-%EB%B2%84%EC%A0%84-%EB%A7%9E%EC%B6%94%EA%B8%B0-%EC%B4%9D%EC%A0%95%EB%A6%AC
  
  
  
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

nvidia-smi
 NVIDIA-SMI 572.61 / Driver Version: 572.61 / CUDA Version: 12.8
nvcc -V
 nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:42:46_Pacific_Standard_Time_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0



11.1 - 11.4  8.6