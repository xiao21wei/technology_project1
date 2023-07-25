import os
import cv2
import numpy as np
from tensorflow import keras
from scipy.spatial.distance import cosine

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def model_test():
    # 加载并预处理ResNet模型
    model = keras.applications.ResNet50(weights='imagenet', include_top=False)

    # 加载目标图像和待搜索图片
    target_image_path = './img/target.jpg'

    # 读取图像
    target_image = cv2.imread(target_image_path)
    # 将图像从BGR格式转换为RGB格式
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    # 调整图像大小为模型输入尺寸
    target_image_resized = cv2.resize(target_image, (224, 224))
    # 预处理图像数据
    target_image_preprocessed = keras.applications.resnet50.preprocess_input(target_image_resized[np.newaxis, :])
    # 提取目标图像的特征
    target_features = model.predict(target_image_preprocessed).flatten()

    for file in os.listdir('./img/picture'):
        image_path = os.path.join('./img/picture', file)
        search_image = cv2.imread(image_path)  # 读取待搜索图片

        # 将图像从BGR格式转换为RGB格式
        search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)

        window_size = (target_image.shape[1], target_image.shape[0])  # 窗口大小与目标图像相同
        stride = 15  # 滑动窗口的步长

        similar_regions = []  # 保存相似度较高的区域

        for (x, y, window) in sliding_window(search_image, window_size, stride):
            # 调整窗口大小为模型输入尺寸
            window_resized = cv2.resize(window, (224, 224))
            window_preprocessed = keras.applications.resnet50.preprocess_input(window_resized[np.newaxis, :])

            # 提取窗口的特征
            window_features = model.predict(window_preprocessed).flatten()
            # 计算特征相似度
            similarity = 1 - cosine(target_features, window_features)
            # 如果相似度较高，则保存窗口位置和相似度
            if similarity > 0.6:
                similar_regions.append((x, y, similarity))

        # 复制待搜索图片并在复制的图片上画矩形框
        search_image_with_boxes = search_image.copy()
        matched_locations = []
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

        # 保存结果
        search_image_with_boxes = cv2.cvtColor(search_image_with_boxes, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./img/result/' + file, search_image_with_boxes)
        pass


def sliding_window(image, window_size, stride):
    for y in range(0, image.shape[0] - window_size[1], stride):
        for x in range(0, image.shape[1] - window_size[0], stride):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


if __name__ == '__main__':
    model_test()
