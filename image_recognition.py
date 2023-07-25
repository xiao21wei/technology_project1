# 从./img/target.jpg中,使用ResNet模型提取特征,
# 与./img/picture/目录下的图片进行比较,找出每张图片中与target相似度较高的部分
# 将相似度较高的部分用矩形框标记出来，并将结果保存在./img/result文件夹中
import copy
import cv2
import numpy as np
from tensorflow import keras
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载库


def image_recognition_test(method, threshold):  # method:匹配方法 threshold:阈值
    target_image_path = './img/target.jpg'

    model = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')  # 加载ResNet模型

    target_image = cv2.imread(target_image_path)  # 读取目标图片

    count = 0  # 计数器,用于记录已处理的图片数量
    for file in os.listdir('./img/picture'):
        image_path = os.path.join('./img/picture', file)
        source_image = cv2.imread(image_path)  # 读取待搜索图片

        # 在待搜索图片中搜索目标图片
        res = cv2.matchTemplate(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), method)

        locations = np.where(res >= threshold)

        # 复制待搜索图片并在复制的图片上画矩形框
        search_image_with_boxes = copy.deepcopy(source_image)
        matched_locations = []
        for loc in zip(*locations[::-1]):  # *号表示可选参数
            top_left = (loc[0], loc[1])  # 左上角,loc[0]为x坐标,loc[1]为y坐标
            bottom_right = (loc[0] + target_image.shape[1], loc[1] + target_image.shape[0])  # 右下角,loc[0]为x坐标,loc[1]为y坐标

            # 检查当前位置是否与已标注的位置重叠,如果重叠则跳过,否则将当前位置标注
            overlapping = False  # 是否重叠
            for prev_top_left, prev_bottom_right in matched_locations:  # 遍历已标注的位置
                if top_left[0] < prev_bottom_right[0] and bottom_right[0] > prev_top_left[0] and \
                        top_left[1] < prev_bottom_right[1] and bottom_right[1] > prev_top_left[1]:
                    overlapping = True
                    break

            if not overlapping:
                count += 1
                cv2.rectangle(search_image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
                matched_locations.append((top_left, bottom_right))

        # 保存结果
        cv2.imwrite('./img/result/' + str(method) + '_' + str(threshold) + '_' + str(file), search_image_with_boxes)
        print('图片' + file + '已保存')

    print('mask:' + str(method) + ' threshold:' + str(threshold) + ' count:' + str(count) + ' OK')


if __name__ == '__main__':
    methods = [cv2.TM_CCOEFF_NORMED]
    thresholds = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    for method in methods:
        for threshold in thresholds:
            image_recognition_test(method, threshold)
