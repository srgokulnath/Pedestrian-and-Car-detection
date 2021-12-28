import cv2

img_file = "pedestrian image.png"
video = cv2.VideoCapture("video1.mp4")
car_classifier_file = "car_detector.xml"
pedestrian_tracker_file = "pedestrian.xml"


# while True:
#     (read_successful, frame) = video.read()
#     if read_successful:
#         grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         break
#     car_tracker = cv2.CascadeClassifier(car_classifier_file)
#     pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)
#     cars = car_tracker.detectMultiScale(grayscaled_frame)
#     pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
#     for (x,y,w,h) in cars:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     for (x, y, w, h) in pedestrians:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


img = cv2.imread(img_file)
car_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars = car_tracker.detectMultiScale(black_n_white) 
# print(cars)
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)


cv2.imshow("carimage", img)
cv2.waitKey()

print("code completed")
