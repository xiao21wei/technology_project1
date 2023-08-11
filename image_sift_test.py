# 使用sift算法提取目标图像的特征，并在待检测图像中进行匹配，使用矩形框标记出目标图像在待检测图像中和目标图像相似度较高的区域
import os
import time

import cv2
import numpy as np


def image_sift_test():
    # 加载目标图像和待搜索图片
    target_image_path = './img/target.jpg'

    # 读取图像
    target_image = cv2.imread(target_image_path)
    # 使用sift算法提取目标图像的特征
    sift = cv2.SIFT_create()
    target_image_kp, target_image_des = sift.detectAndCompute(target_image, None)

    data = []
    for file in os.listdir('./img/picture'):
        image_path = os.path.join('./img/picture', file)
        search_image = cv2.imread(image_path)

        window_size = (target_image.shape[1], target_image.shape[0])  # 窗口大小与目标图像相同
        stride = 15  # 滑动窗口的步长

        similar_regions = []  # 保存相似度较高的区域

        for (x, y, window) in sliding_window(search_image, window_size, stride):
            # 使用sift算法提取窗口图像的特征
            window_kp, window_des = sift.detectAndCompute(window, None)
            # 使用FLANN算法进行匹配
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(target_image_des, window_des, k=2)
            # 计算匹配的特征点
            good = []
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append(m)
            # 如果匹配的特征点数大于10，则保存窗口位置和匹配的特征点数
            if len(good) > 120:
                similar_regions.append((x, y, len(good)))

        # 复制待搜索图片并在复制的图片上画矩形框
        search_image_with_boxes = search_image.copy()
        matched_locations = []
        similarities = []
        for x, y, similarity in similar_regions:
            top_left = (x, y)
            bottom_right = (x + target_image.shape[1], y + target_image.shape[0])

            # 检查当前位置是否与已标注的位置重叠,如果重叠则跳过,否则将当前位置标注
            overlapping = False
            for prev_top_left, prev_bottom_right in matched_locations:
                if top_left[0] < prev_bottom_right[0] and bottom_right[0] > prev_top_left[0] and \
                        top_left[1] < prev_bottom_right[1] and bottom_right[1] > prev_top_left[1]:
                    # overlapping = True
                    break

            if not overlapping:
                cv2.rectangle(search_image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
                print(f'Found similar region with similarity {similarity} at {top_left}')
                matched_locations.append((top_left, bottom_right))
                similarities.append(similarity)

        # 保存标注了矩形框的图片
        cv2.imwrite('./img/result/' + file, search_image_with_boxes)
        data.append(similarities)
    print(data)
    pass


def sliding_window(image, window_size, stride):
    for y in range(0, image.shape[0] - window_size[1], stride):
        for x in range(0, image.shape[1] - window_size[0], stride):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


if __name__ == '__main__':
    time1 = time.time()
    image_sift_test()
    time2 = time.time()
    print('time cost:', time2 - time1, 's')
