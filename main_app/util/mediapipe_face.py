import cv2
import mediapipe as mp
import imutils


def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def check_good_eye(x_left, x_right):
    limit = 0.5
    if (x_right > limit) or (x_left < 1 - limit):
        return False
    return True


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0)
# For static images:


def check_face_mp(image, width, min_confidence=0.8):
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    image = imutils.resize(image, width=width)
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
        return 1, 0

    detection = sorted(results.detections,
                       key=lambda x: x.score[0], reverse=True)[0]
    conf = detection.score[0]
    if conf < min_confidence:
        return 1, 0
    left_eye = mp_face_detection.get_key_point(
        detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
    right_eye = mp_face_detection.get_key_point(
        detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

    x_left_eye_raito = left_eye.x
    # y_left_eye = int(left_eye.y * image.shape[0])
    x_right_eye_raito = right_eye.x
    # y_right_eye = int(right_eye.y * image.shape[0])
    # cv2.circle(image, (int(x_left_eye_raito *
    #            image.shape[1]), int(left_eye.y * image.shape[0])), 5, (0, 0, 255), -1)
    # cv2.circle(image, (int(x_right_eye_raito *
    #            image.shape[1]), int(right_eye.y * image.shape[0])), 5, (0, 0, 255), -1)
    is_good_eye = check_good_eye(x_left_eye_raito, x_right_eye_raito)
    if not is_good_eye:
        return 1, 0
    return 0, conf
