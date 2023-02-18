import cv2


def is_in_polygon(point, polygon):
    x, y = point
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def count_object(polygon, old_id_dict, id_dict):
    car_count = 0
    motor_count = 0
    for id, bbox in old_id_dict.items():
        point = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        if not is_in_polygon(point, polygon):
            continue
        if id in id_dict.keys():
            bbox = id_dict[id]
            cls = id_dict[id][4]
            point_2 = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            if not is_in_polygon(point_2, polygon):
                if cls == 0.0:
                    car_count += 1
                elif cls == 1.0:
                    motor_count += 1
    return car_count, motor_count
