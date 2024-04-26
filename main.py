import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取要拼接的图片
img1 = cv2.imread('imgs/image5.jpg')
img2 = cv2.imread('imgs/image6.jpg')
imageA = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2)
imageB = cv2.resize(img2, (0, 0), fx=0.2, fy=0.2)

#创建sift对象
sift = cv2.SIFT_create()
#计算关键点，描述符
kp1, descriptors1  = sift.detectAndCompute(imageA, None)
kp2, descriptors2  = sift.detectAndCompute(imageB, None)

#进行两图的特征匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)


#计算透视变换矩阵
if len(good) > 20:
    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ano_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_points, ano_points, cv2.RANSAC, 5.0)
    warpImg = cv2.warpPerspective(imageB, np.linalg.inv(M), (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
    result = warpImg.copy()
    result[0:imageA.shape[0], 0:imageB.shape[1]] = imageA

rows, cols = imageA.shape[:2]


#找到两张图片的交界线
left_col = np.argwhere(np.any(imageA, axis=0)).min()
right_col = np.argwhere(np.any(imageA, axis=0)).max()

left = left_col.item()
right = right_col.item()

#将图像A和变换后的图像B进行混合
res = np.zeros([rows, cols, 3], np.uint8)
for row in range(0, rows):
    for col in range(0, cols):
        if not imageA[row, col].any():
            res[row, col] = warpImg[row, col]
        elif not warpImg[row, col].any():
            res[row, col] = imageA[row, col]
        else:
            srcImgLen = float(abs(col - left))
            testImgLen = float(abs(col - right))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            res[row, col] = np.clip(imageA[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
warpImg[0:imageA.shape[0], 0:imageA.shape[1]] = res
warpImg[0:imageA.shape[0], 0:imageA.shape[1]] = res


#处理黑边
gray_image = cv2.cvtColor(warpImg, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
# 使用 findContours 找到非背景区域的轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 计算非背景区域的边界矩形
x_min, y_min = np.inf, np.inf
x_max, y_max = -np.inf, -np.inf
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    x_min, y_min = min(x, x_min), min(y, y_min)
    x_max, y_max = max(x + w, x_max), max(y + h, y_max)

# 根据边界矩形裁剪图像，为了去掉黑色背景让图片显示为长方形
cropped_image3 = warpImg[y_min:y_max, x_min:x_max]

cv2.imshow('Cropped result', cropped_image3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_imgs/cropped_image.jpg',cropped_image3)