import cv2
import numpy as np
import time
import math


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def houghlines(img, threshold):
    # initialize Rho and Theta ranges
    lines = np.empty((1, 4), int)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, num=diag_len * 2.0,
                       dtype=int)  # Return evenly spaced numbers over a specified interval.
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    # only pixels that non-zero are edges:
    yIndex, xIndex = np.nonzero(img)  # (row, col) indexes to edges
    XMin = np.full(accumulator.shape, np.argmax(xIndex))
    YMax = np.full(accumulator.shape, np.argmax(yIndex))
    XMax = np.zeros(accumulator.shape)
    YMin = np.zeros(accumulator.shape)

    for i in range(len(xIndex)):
        x = xIndex[i]
        y = yIndex[i]
        for ThetaIndex in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[ThetaIndex] + y * sin_t[ThetaIndex]))
            accumulator[rho, ThetaIndex] += 1
            if y > YMin[rho, ThetaIndex]:
                YMin[rho, ThetaIndex] = y
                XMin[rho, ThetaIndex] = x
            else:
                if y < YMax[rho, ThetaIndex]:
                    YMax[rho, ThetaIndex] = y
                    XMax[rho, ThetaIndex] = x
    for h in range(accumulator.shape[0]):
        for w in range(accumulator.shape[1]):
            if accumulator[h, w] > threshold:  # filtering by threshold
                x1 = XMin[h, w]
                y2 = YMax[h, w]
                x2 = XMax[h, w]
                y1 = YMin[h, w]
                lines = np.vstack((lines, np.array([x1, y1, x2, y2], int)))
    return lines


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if (abs(y1 - y2) >= 10) and (abs(x1 - x2) <= abs(y1 - y2) * 2.5):
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = edgeDetectionCanny(image, lower, upper)
    # return the edged image
    return edged


def edgeDetectionCanny(image, thrs_1, thrs_2):
    I = (image * 255).astype(np.uint16)
    thrs_1 *= .75
    thrs_2 *= .75
    mag, ori = getSobelMagOri(I)
    quant_ori = np.degrees(ori)
    quant_ori[quant_ori < 0] += 180
    quant_ori_x = np.zeros(quant_ori.shape, dtype=np.int)
    quant_ori_y = np.zeros(quant_ori.shape, dtype=np.int)
    # angle 0
    quant_ori_x[(0 <= quant_ori) & (quant_ori < 22.5) | (157.5 <= quant_ori) & (quant_ori < 180)] += 1

    # angle 45
    quant_ori_x[(22.5 <= quant_ori) & (quant_ori < 67.5)] += 1
    quant_ori_y[(22.5 <= quant_ori) & (quant_ori < 67.5)] -= 1

    # angle 90
    quant_ori_y[(67.5 <= quant_ori) & (quant_ori < 112.5)] += 1

    # angle 135
    quant_ori_x[(112.5 <= quant_ori) & (quant_ori < 157.5)] += 1
    quant_ori_y[(112.5 <= quant_ori) & (quant_ori < 157.5)] += 1

    h, w = I.shape[:2]
    mag_c = mag.copy()
    for iy in range(1, h - 1):
        for ix in range(1, w - 1):
            grad_x = quant_ori_x[iy, ix]
            grad_y = quant_ori_y[iy, ix]
            v = mag[iy, ix]
            pre = mag[iy - grad_y,
                      ix - grad_x]
            post = mag[iy + grad_y,
                       ix + grad_x]

            if v <= pre or post >= v:
                mag_c[iy, ix] = 0
    thrs_map_2 = (mag_c >= thrs_2).astype(np.uint8)
    thrs_map_2_dilate = cv2.dilate(thrs_map_2, np.ones((2, 2)))
    thrs_map_1 = ((mag_c < thrs_2) & (mag_c >= thrs_1)).astype(np.uint8)

    my_canny = (thrs_map_2 | (thrs_map_1 & thrs_map_2_dilate))
    my_canny *= 255
    return my_canny


def getSobelMagOri(I):
    sobel_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]).astype(np.float64)
    i_x = conv2D(I, sobel_kernel)
    i_y = conv2D(I, -sobel_kernel.T)
    mag = np.hypot(i_x, i_y)
    ori = np.arctan2(i_y, i_x)
    return mag, ori


def conv2D(inImage, kernel2):
    inv_k = kernel2[::, ::]
    kernel_shape = np.array([x for x in kernel2.shape])
    img_shape = np.array([x for x in inImage.shape])
    out_len = np.max([kernel_shape, img_shape], axis=0)
    midKernel = kernel_shape // 2
    paddedSignal = np.pad(inImage.astype(np.float32),
                          ((kernel_shape[0], kernel_shape[0]),
                           (kernel_shape[1], kernel_shape[1]))
                          , 'edge')
    outSignal = np.ones(out_len)
    for i in range(out_len[0]):
        for j in range(out_len[1]):
            st_x = j + midKernel[1] + 1
            end_x = st_x + kernel_shape[1]
            st_y = i + midKernel[0] + 1
            end_y = st_y + kernel_shape[0]
            outSignal[i, j] = (paddedSignal[st_y:end_y, st_x:end_x] * inv_k).sum()
    return outSignal


def histeq(im):
    # calculate Histogram
    h = imhist(im)
    cdf = np.array(cumsum(h))  # cumulative distribution function
    sk = np.uint8(255 * cdf)  # finding transfer function values
    s1, s2 = im.shape
    Y = np.zeros_like(im)
    # applying transfered values for each pixels
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    return Y


def imhist(im):
    # calculates normalized histogram of an image
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h) / (m * n)


def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i + 1]) for i in range(len(h))]


def process(image):
    if image is not None:
        print(image.shape)
        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (0, height),
            (width * 0.43, height * 0.55),
            (width * 1.3, height)
        ]
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalizeHist_image = histeq(gray_image)
        #cv2.imshow('equalizeHist_image', equalizeHist_image)
        canny_image = auto_canny(equalizeHist_image)
        #cv2.imshow('canny_image', canny_image)
        cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32), )
        #cv2.imshow('cropped_image', cropped_image)
        lines = houghlines(cropped_image, threshold=100)
        image_with_lines = drow_the_lines(image, lines)
        return image_with_lines
    else:
        return image


cap = cv2.VideoCapture('Car_Vid.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    start = time.clock()
    frame = process(frame)
    elapsedTime = time.clock()-start
    print("process took %2.2f secs" %elapsedTime)
    if ret == True:
        height, width, channels = frame.shape
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
