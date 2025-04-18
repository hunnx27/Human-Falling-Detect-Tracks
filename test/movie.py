import cv2

#video = cv2.VideoCapture("./test/falldown.mp4")
video = cv2.VideoCapture("./downloaded_video.mp4")

print(video)
print('### before')
print(video.isOpened())
while video.isOpened():
    check, frame = video.read()
    if not check:
        print("Frame이 끝났습니다.")
        break

    cv2.imshow("cute cats",frame)
    if cv2.waitKey(25) == ord('q'):
        print("동영상 종료")
        break

print('### after')
video.release()
cv2.destroyAllWindows()