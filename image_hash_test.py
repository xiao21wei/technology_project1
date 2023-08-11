import os
import cv2
import time
import numpy as np
from scipy.spatial.distance import hamming

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def image_hash(target_image):
    # 将图像转换为16*16
    target_image = cv2.resize(target_image, (16, 16))
    # 将图像转换为灰度图
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    # 计算图像的平均灰度
    average = np.mean(target_image)
    # 将图像转换为二值图
    target_image = (target_image > average) * 1.00
    # 将二值图转换为哈希值
    target_image_hash = []
    for row in target_image.tolist():
        for i in row:
            target_image_hash.append(int(i))
    print(target_image_hash)
    return target_image_hash
    pass


def model_hash_test():
    # 加载目标图像和待搜索图片
    target_image_path = './img/target.jpg'

    # 读取图像
    target_image = cv2.imread(target_image_path)
    # 计算图像的哈希值
    target_image_hash = image_hash(target_image)

    data = []
    for file in os.listdir('./img/picture'):
        image_path = os.path.join('./img/picture', file)
        search_image = cv2.imread(image_path)  # 读取待搜索图片

        window_size = (target_image.shape[1], target_image.shape[0])  # 窗口大小与目标图像相同
        stride = 10  # 滑动窗口的步长

        similar_regions = []  # 保存相似度较高的区域

        for (x, y, window) in sliding_window(search_image, window_size, stride):
            # 计算窗口的哈希值
            window_hash = image_hash(window)
            # 计算哈希值的汉明距离,并计算相似度,汉明距离越小,相似度越高
            similarity = round(1 - hamming(target_image_hash, window_hash), 8)
            # 如果相似度较高，则保存窗口位置和相似度
            if similarity > 0.7:
                similar_regions.append((x, y, similarity))

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
                    overlapping = True
                    break

            if not overlapping:
                cv2.rectangle(search_image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
                print(f'Found similar region with similarity {similarity} at {top_left}')
                matched_locations.append((top_left, bottom_right))
                similarities.append(similarity)

        # 保存结果
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
    model_hash_test()
    time2 = time.time()
    print('总共耗时：' + str(time2 - time1) + 's')
