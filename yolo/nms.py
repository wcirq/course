import numpy as np
import cv2


def draw(matrix, win_name):
    scale = 1
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    for x1, y1, x2, y2, score in matrix:
        x1 = int(x1*scale)
        y1 = int(y1*scale)
        x2 = int(x2*scale)
        y2 = int(y2*scale)
        color = list(int(i) for i in np.random.randint(0, 255, (3,)))
        cv2.putText(canvas, f"{score}", (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color)
    cv2.imshow(win_name, canvas)


def cal_nms(matrix, thresh):
    """NMS算法"""
    x1 = matrix[:, 0]
    y1 = matrix[:, 1]
    x2 = matrix[:, 2]
    y2 = matrix[:, 3]
    scores = matrix[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == '__main__':
    # x1, y1, x2, y2, c
    matrix = np.array([[310, 5, 420, 120, 0.6],
                     [20, 20, 240, 210, 1],
                     [70, 50, 260, 220, 0.8],
                     [400, 280, 560, 360, 0.7],
                     [380, 276, 500, 370, 0.8]
                       ])
    # 绘制
    draw(matrix, "matrix")
    keep = cal_nms(matrix, 0.4)
    keep_matrix = matrix[keep]
    draw(keep_matrix, "keep_matrix")
    print(keep)
    cv2.waitKey(0)