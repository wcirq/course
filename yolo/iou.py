def cal_iou(box1, box2):
    """计算box1和box2的交并比（IoU）
    Arguments:
    box1 -- 左上角和右下角的坐标 (x1, y1, x2, y2)
    box2 -- 左上角和右下角的坐标 (x1, y1, x2, y2)
    """
    # 获取两个框相交部分的矩形的左上角和右下角坐标
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    # 计算两个框相交部分的面积
    inter_area = (yi2 - yi1) * (xi2 - xi1)
    # 分别计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 计算两个框的并集部分面积
    union_area = box1_area + box2_area - inter_area
    # 计算交并比
    iou = inter_area / union_area
    return iou


if __name__ == '__main__':
    # (x1, y1, x2, y2)
    box1 = [0, 0, 2, 2]
    box2 = [1, 1, 3, 3]
    iou = cal_iou(box1, box2)
    print("IOU", iou)
