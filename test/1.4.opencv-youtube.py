"""
pip install pyutbefix
"""

import cv2
#from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress

# # YouTube 동영상 URL
video_url = 'https://www.youtube.com/watch?v=f58k04x942M'
# # YouTube 객체 생성
yt = YouTube(video_url, on_progress_callback =on_progress)
# # 가장 높은 품질의 비디오 스트림 선택
#video_stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
video_stream = yt.streams.get_highest_resolution()
# # 비디오 다운로드 경로 설정
download_path = 'downloaded_video.mp4'
# # 다운로드 시작
video_stream.download(filename=download_path)
video = cv2.VideoCapture(download_path)

while video.isOpened():
    check, frame = video.read()
    if not check:
        print("Frame이 끝났습니다.")
        break

    cv2.imshow("cute cats",frame)
    if cv2.waitKey(25) == ord('q'):
        print("동영상 종료")
        break

video.release()
cv2.destroyAllWindows()