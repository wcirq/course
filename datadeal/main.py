# -*- coding: utf-8 -*- 
# @File main.py
# @Time 2021/3/15 16:37
# @Author wcy
# @Software: PyCharm
# @Site
import os
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

classes = ["text"]


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, max_boxes=100):
    '''处理图片和标签'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32) / 255

    # 处理 boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        # 将box实际标签的左上角和右下角坐标转为图片宽高为（nw, nh)时的值
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        # 将box坐标为负值的强制转为0
        box[:, 0:2][box[:, 0:2] < 0] = 0
        # 将box坐标超过图片宽高的强制转为对应的宽高
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        # 重新计算box宽高
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # 丢弃无效 box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors) // 3
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    #   获得框的坐标和图片的大小
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    #   通过计算获得真实框的中心和宽高
    #   中心点(m,n,2) 宽高(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #   将真实框归一化到小数形式
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    #   [9,2] -> [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    #   长宽要大于0才有效
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue

        #   [n,2] -> [n,1,2]

        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        #   计算所有真实框和先验框的交并比
        #   intersect_area  [n,9]
        #   box_area        [n,1]
        #   anchor_area     [1,9]
        #   iou             [n,9]
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        #   维度是[n,]
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            #   找到每个真实框所属的特征层
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')

                    #   k指的的当前这个特征点的第k个先验框
                    k = anchor_mask[l].index(n)
                    #   c指的是当前这个真实框的种类
                    c = true_boxes[b, t, 4].astype('int32')
                    #   y_true的shape为(m,13,13,3,6)(m,26,26,3,6)(m,52,52,3,6)
                    #   最后的85可以拆分成4+1+1，4代表的是框的中心与宽高、
                    #   1代表的是置信度、1代表的是种类
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):
    while True:
        np.random.shuffle(annotation_lines)
        image_data = []
        box_data = []
        for b in range(batch_size):
            image, box = get_random_data(annotation_lines[b], input_shape)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true]


def convert_annotation(file_name):
    image_id = os.path.splitext(file_name)[0]
    in_file = open(f'data/annotations/{image_id}.xml', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    results = f"data/images/{image_id}.jpg"
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        results += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
    return results


def main():
    batch_size = 3
    num_classes = len(classes)
    input_shape = (416, 416)
    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    anchors = np.array(anchors).reshape(-1, 2)
    images_file_list = os.listdir(f"data/images")
    annotation_lines = [convert_annotation(file_name) for file_name in images_file_list]
    datasets = data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True)
    for data in datasets:
        images = data[0]
        output13_13 = data[1]
        output26_26 = data[2]
        output52_52 = data[3]
        print()


if __name__ == '__main__':
    main()
