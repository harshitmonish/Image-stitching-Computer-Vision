# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:13:52 2021

@author: harshit monish
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random
import matplotlib.pyplot as plt

# Function to calulate the Homography matrix
def calculate_homography_matrix(data):
    a_list = []
    for point in data:
        # creating the a matrix and adding each x, y values from each image
        r1 = [point.item(2), point.item(3), 1, 0, 0, 0, -(point.item(0)*point.item(2)), -(point.item(0)*point.item(3)),
              -(point.item(0)) ]
                                                
        r2 = [0, 0, 0, point.item(2), point.item(3), 1, -(point.item(1)*point.item(2)), - (point.item(1)*point.item(3)), 
              -point.item(1)]
        a_list.append(r1)
        a_list.append(r2)
    
    # A matrix
    a_mat = np.array(a_list)

    u, s, v = np.linalg.svd(a_mat)
    
    # reshaping the min row of v into 3x3 matrix
    h = np.reshape(v[8], (3, 3))
    
    #normalizing 
    h = (1/(h.item(8)))*h
    
    return h
    

# function to get the distance for calculating inliners    
def get_distance(samples, h):
    temp1 = np.transpose(np.matrix([samples[0].item(0), samples[0].item(1), 1]))
    temp2 = np.dot(h, temp1)
    temp2 = (1/temp2.item(2))*temp2
    
    temp3 = np.transpose(np.matrix([samples[0].item(2), samples[0].item(3), 1]))
    error = temp3 - temp2
    res = np.linalg.norm(error)
    return res

# Ransac Implementation
def ransac_func(iters, t, num_samples, data):
    maxInliners = []
    h_matrix = None
    itr = 0
    while itr != iters:
        if(itr%100 == 0):
            print("Iterstion: "+str(itr))
            
        # setting seed value for consistent results
        random.seed(10)
        
        # choosing 4 points at random
        s1 = data[random.randrange(0, len(data))]
        s2 = data[random.randrange(0, len(data))]
        samples = np.vstack((s1, s2))
        s3 = data[random.randrange(0, len(data))]
        samples = np.vstack((samples, s3))
        s4 = data[random.randrange(0, len(data))]
        samples = np.vstack((samples, s4))
        
        # computing the homography matrix for points choosen
        h = calculate_homography_matrix(samples)
        inliners = []
        
        # calculating the inliner by computing the distance 
        for i in range(len(data)):
            distance = get_distance(data[i], h)
            if(distance < t):
                inliners.append(data[i])
                
        # Saving maximum inliners
        if(len(inliners) > len(maxInliners)):
            maxInliners = inliners
            h_matrix = h
            
        itr+=1
    if(h_matrix is None):
        h_matrix = h
    return h_matrix, maxInliners

# function to warp and stitch the images together
def wrap_and_stitch_img(left_img, right_img, h):
    
    # Getting dimensions of both the images
    r1 = left_img.shape[0]
    c1 = left_img.shape[1]
    r2 = right_img.shape[0]
    c2 = right_img.shape[1]
    
    points_list1 = np.float32([[0, 0], [0, r1], [c1, r1], [c1, 0]]).reshape(-1, 1, 2)
    points_list2 = np.float32([[0, 0], [0, r2], [c2, r2], [c2, 0]]).reshape(-1, 1, 2)
    
    # getting the dimensions for panoramic view window
    points_list3 = cv2.perspectiveTransform(points_list2, h)
    
    points = np.concatenate((points_list1, points_list3), axis=0)
    [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
    
    # Creating the transformed h matrix
    tran_dist = [-xmin, -ymin]
    h_tran = np.array([[1, 0, tran_dist[0]], [0, 1, tran_dist[1]], [0, 0, 1]])
    
    # Using the transformed h matrix to warp the right image in the panoramic view window
    # And adding the left image in it.
    res_img = cv2.warpPerspective(right_img, h_tran.dot(h), (xmax-xmin, ymax - ymin))
    res_img[tran_dist[1]:r1+tran_dist[1], tran_dist[0]:c1+tran_dist[0]] = left_img

    return res_img
 
# Function to implement KNN for feature mapping in left and right images    
def knn_match(desl, kpl, desr, kpr, k):
    matches = []
    for i in range(desl.shape[0]):
        x = desl[i]
        distances = []
        distance_list = np.linalg.norm(x-desr, axis=1) 

        dist_index = np.array(distance_list).argsort()[:k]

        if(distance_list[dist_index[0]] < 0.6 * distance_list[dist_index[1]]):
            (x1, y1) = kpl[i].pt
            (x2, y2) = kpr[dist_index[0]].pt
            matches.append([x1, y1, x2, y2])
    return matches

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    sift = cv2.xfeatures2d.SIFT_create()
    
    print("\n Extracting Features from both images")
    # Getting Key points and Descriptors for left and right image
    kpl, desl = sift.detectAndCompute(left_img, None)
    kpr, desr = sift.detectAndCompute(right_img, None)
       
    print("\n Feature Matching using KNN with k=2 ")
    matches = knn_match(desl, kpl, desr, kpr, 2)

    points = np.matrix(matches)
    
    
    print("Running Ransac Algorithm:")
    h, inliners = ransac_func(1000, 6, 4, points)
    
    print("\n Wrapping and Stitching the images")
    final_img = wrap_and_stitch_img(left_img, right_img, h)

    result_img = final_img
    return result_img
  
    
if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')

    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


