# importing some useful packages

import numpy as np
import cv2


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def region_of_interest(img):
    # calculating the height of an image
    height = img.shape[0]
    width = img.shape[1]

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a polygon as region of interest
    lowerLeftPoint = [130, 540]
    upperLeftPoint = [410, 350]
    upperRightPoint = [570, 350]
    lowerRightPoint = [915, 540]

    polygon = np.array([
        [lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]])

    # fill(or addition of) the mask with the polygon
    cv2.fillPoly(mask, polygon, 255)

    # masking the image with the mask created mask using bitwise and
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def display_lines(image, lines):
    mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return mask


# # reading the input
# image = cv2.imread('test_images/test.jpeg')
# lane_image = np.copy(image)


cap = cv2.VideoCapture('test_videos/test.mp4')
while (cap.isOpened()):
    _, frame = cap.read()
    # grayscale the image
    grayscaled = grayscale(frame)

    # apply gaussian blur
    kernelSize = 5
    gaussianBluredImg = gaussian_blur(grayscaled, kernelSize)

    # canny edge detection
    minThreshold = 50
    maxThreshold = 150
    edgeDetectedImage = canny(gaussianBluredImg, minThreshold, maxThreshold)

    # finding masked image
    masked_image = region_of_interest(edgeDetectedImage)

    # determining the lines using hough probablistic transform
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)

    # creating black images with the lines using hough probablistic transform
    line_image = display_lines(frame, lines)

    # adding the masked_image and line_image
    lined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    ###
    cv2.imshow("result", lined_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.distroyAllWindows()
