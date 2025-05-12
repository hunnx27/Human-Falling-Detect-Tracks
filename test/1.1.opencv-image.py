import cv2

image = cv2.imread("image.png",cv2.IMREAD_UNCHANGED)
#image = cv2.imread("sample1.gif",cv2.IMREAD_UNCHANGED)
print(image)
print(type(image)) # numpy.ndarray : # 다차원 행렬 자료구조 클래스
cv2.imshow("Universe",image) # 제목넣기
cv2.waitKey(0) # 무한대기 - 어떠한키가 들어와도 종료됨.(키가 들어올때 까지 blocking 상태로 대기)
cv2.destroyAllWindows() # 모든창을 닫음