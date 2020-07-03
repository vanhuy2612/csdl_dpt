# cv2.calHist : mỗi giá trị tương ứng với số pixel trong hình ảnh đó với giá trị pixel tương ứng.
# np.histogram :  thay vì tính tổng số pixel thì nó lại tính tổng giá trị trong các pixel (weights)
import os
import cv2
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt

# link data
city_path = './data/city'
forest_path = './data/forest'
sea_path = './data/sea'
# link image
city_image_path = [os.path.join(city_path,i) for i in os.listdir(city_path)]
forest_image_path = [os.path.join(forest_path,i) for i in os.listdir(forest_path)]
sea_image_path = [os.path.join(sea_path,i) for i in os.listdir(sea_path)]

size = 100
test_image_path = city_image_path[480:] + forest_image_path[480:] + sea_image_path[480:]
train_image_path = city_image_path[:size] + forest_image_path[:size] + sea_image_path[:size]


# Hàm trích xuất histogram theo màu sắc (dùng hệ màu HSV):
def fd_histogram(image_path, mask=None, bins=10):
    image = cv2.imread(image_path)
    image = cv2.resize(src=image, dsize=(256, 256))
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tính histogram của ảnh thông qua 3 kênh HSV biểu thị tổng giá trị khối lượng 
    # HSV (Hue – vùng màu, Saturation – độ bão hòa màu, Value – độ sáng).
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) → hist
    # 1 phần tử của ma trận hist ở vị trí [2, 4, 5] : biểu thị số pixel có : 
    # { 
    #   giá trị H thuộc bins 2 trong giải giá trị của H
    #   giá trị S thuộc bins 4 trong giải giá trị của S
    #   giá trị V thuộc bins 5 trong giải giá trị của V
    # }  
    hist  = cv2.calcHist([image], channels=[0, 1, 2], mask=None, histSize=[bins, bins, bins], ranges=[0, 256, 0, 256, 0, 256])
    # normalize the histogram : chuẩn hóa ma trận thành dạng L2-Norm : độ dài =1
    # chuẩn hóa trên cả ma trận hist:
    print(hist)
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
   
    # return the histogram
    return hist.flatten() # bins*bins*bins = 1000 features with bins = 10
#fd_histogram(train_image_path[0])
# trích xuất màu sắc trong hệ màu RGB
def fd_histogram_v2(image_path):
    image = cv2.imread(image_path)
    # chuyển lại kích thước ảnh :
    image = cv2.resize(src=image, dsize=(256, 256))
    # Chuyển đổi hình ảnh sang 1 không gian màu RGB.
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # print(image)
    # print(image.shape)
    #plt.imshow(image)
    #plt.show()
    # Sử dụng để phân chia một hình ảnh thành ba mảng cường độ khác nhau cho mỗi kênh màu,
    chans = cv2.split(image) 
    colors = ("B", "G", "R") # thứ tự chuẩn.
    
    features = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256]) # 256 bins
        features.extend(hist)
    features_vector = np.array(features).flatten() # 
    # normalize the histogram : chuẩn hóa ma trận thành dạng L2-Norm : độ dài =1
    # chuẩn hóa trên cả ma trận features_vector:
    print(features_vector)
    cv2.normalize(features_vector, features_vector, norm_type=cv2.NORM_L2)

    # print(features_vector)
    # print(features_vector.shape)
    return features_vector # 768 x 1 with bins of each channel = 256
# print(fd_histogram_v2(train_image_path[0]))
#fd_histogram_v2(train_image_path[0])
# Hàm trích xuất theo thuật toán HOG :
def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape # height=128, width=64
    
    # gradient
    # cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) → dst
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel) # Calculate the x and y gradients size không đổi
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)
    
    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx+0.00001)) # radian
    orientation = np.degrees(orientation) # -90 -> 90
    orientation += 90 # 0 -> 180
    
    num_cell_x = w // cell_size # 8
    num_cell_y = h // cell_size # 16
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins]) # 16 x 8 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            mag = magnitude[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
            # Tính histrogram theo tổng giá trị khối lượng trong các bins:
            # numpy.histogram(input, bins=10, range=None, normed=None, weights=None, density=None)[source]
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag) # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass
    
    # normalization : chuẩn hóa trên block
    redundant_cell = block_size-1
        # feature_tensor # 7 x 15 x ( 2 x 2 x 9) = 7 x 15 x 36
    feature_tensor = np.zeros([num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
    for bx in range(num_cell_x-redundant_cell): # 7
        for by in range(num_cell_y-redundant_cell): # 15
            by_from = by
            by_to = by+block_size
            bx_from = bx
            bx_to = bx+block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten() # to 1-D array (vector)
            # chuẩn hóa vector v thành dạng L2-norm: độ dài vector v = 1
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any(): # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v
    
    return feature_tensor.flatten() # 3780 features
# Chuyển đặc toàn bộ đặc trưng của ảnh thành 1 vector: 3780(hình dạng) + 1000 (màu sắc)
def convert_to_vector(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(64, 128)) # 
    plt.imshow(img)
    plt.show()
    # đặc trưng hình dạng
    f = hog(img)
    # đặc trưng màu sắc dùng version 2
    color_features = fd_histogram_v2(img_path)
    result = np.append([], f)
    result = np.append(result, color_features) 
    return result
#convert_to_vector(train_image_path[1])
# Hàm tính khoảng cách giữa 2 vector bằng Ln_Norm
def distance(v1,v2):
    total = 0
    for i in range(len(v1)):
        # using L2_Norm
        total += (v1[i] - v2[i])*(v1[i] - v2[i])
    return math.sqrt(total)

# Hàm tính độ tương đồng giữa 2 vector 
def distance_v2(v1, v2):
    total = 0
    for i in range(len(v1)):
        total += v1[i]*v2[i]
    return total
# Lấy list feature từ ảnh training
def getListFeature(train_image_path):
    global_features = []
    for i in range( len(train_image_path)):
        global_features.append(convert_to_vector(train_image_path[i]))
    return global_features
# Hàm tìm nhãn cho ảnh đầu vào bằng thước đo giữa 2 bản ghi Ln_Norm
def find_label(global_features, image_path):
    # show Image
    print("Ảnh đầu vào:")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(64, 128))
    plt.imshow(img)
    plt.show()
    # convert image to vector
    imageVt = convert_to_vector(image_path)
    
    min = 10000
    index = -1
    # find min and index of distance
    for i in range(len(global_features)):
        if min >= distance(global_features[i], imageVt):
            min = distance(global_features[i], imageVt)
            index = i
    # fine label for image
    print("distance_min:")
    print(min)
    print("index_min:")
    print(index)
    print("ảnh giống nhất:")
    image = cv2.imread(train_image_path[index], cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(src=image, dsize=(64, 128))
    plt.imshow(image)
    plt.show()
    if index <= size-1:
        print("this is city")
    elif index <= 2*size-2:
        print("this is forest")
    else : 
        print("this is sea")

# Hàm tìm nhãn cho ảnh đầu vào bằng truy vấn không gian vector phụ thuộc vào sự tương đồng
def find_label_v2(global_features, image_path):
    # show Image
    print("Ảnh đầu vào:")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(64, 128))
    plt.imshow(img)
    plt.show()
    # convert image to vector
    imageVt = convert_to_vector(image_path)
    
    max = -1
    index = -1
    # find min and index of distance
    for i in range(len(global_features)):
        if max <= distance_v2(global_features[i], imageVt):
            max = distance_v2(global_features[i], imageVt)
            index = i
    # fine label for image
    print("tương đồng lớn nhất:")
    print(max)
    print("index_max:")
    print(index)
    print("ảnh giống nhất:")
    image = cv2.imread(train_image_path[index], cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(src=image, dsize=(64, 128))
    plt.imshow(image)
    plt.show()
    if index <= size-1:
        print("this is city")
    elif index <= 2*size-2:
        print("this is forest")
    else : 
        print("this is sea")
# Bắt đầu chạy code:
global_features = [] # mảng các feature được trích xuất ra
global_features = getListFeature(train_image_path)
find_label_v2(global_features, test_image_path[1])
