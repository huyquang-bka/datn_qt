import cv2

points = []


def get_mouse_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


path = r"rtmp://192.168.1.197:60005/live/2"
cap = cv2.VideoCapture(path)
image = cap.read()[1]
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_click)

while True:
    image_copy = image.copy()
    for point in points:
        cv2.circle(image_copy, tuple(point), 3, (0, 0, 255), -1)
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(image_copy, tuple(points[i]), tuple(
                points[i + 1]), (0, 255, 0), 2)
    cv2.imshow("image", image_copy)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("c"):
        if points:
            points.remove(points[-1])
    if key == ord("s"):
        print(points)
        break
