# -*- coding: utf-8 -*-
"""
learning where the center of the target is
Created on Thu Dec 03 12:36:48 2015

@author: borgr
"""
import util
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import PIL.Image
THRESHOLD = 100
TRAIN_IMAGE = r'C:\Users\borgr\Google Drive\autofly-510\database\yes\yes5\20151181442_0003465.jpg'
DIR_PATH = util.YES_SMALL_TEST
#COMPARE_IMAGE = r'C:\Users\borgr\Google Drive\autofly-510\database\yes\yes5\20151181442_0003655.jpg'
JSON_FILE = '.\\JsonCenters'
KP_BF_JASON = r'.\kp_find_ten_threshold100.json'
DIRS = "dir"  # key of the finished directories list


def compare_images(path1, path2):
    """
    path1 - train image
    compares 2 images and returns the best matcher"""
    img_test = cv2.imread(path1, 0)
    img_query = cv2.imread(path2, 0)
    surf = cv2.SURF(THRESHOLD)

    #  create features
    kp_test, des_test = surf.detectAndCompute(img_test, None)
    kp_query, des_query = surf.detectAndCompute(img_query, None)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    if len(des_query) > 10:
        matches = bf.knnMatch(des_test[1:2], des_query, k=10)
        # Sort them in the order of their distance.
        matches = sorted(matches[0], key=lambda x: x.distance)
        return [kp_query[matches[i].queryIdx].pt for i in range(10)]
    print (path2)
    return None


def create_file(dir_path, file_name, to_print=set(util.PRINT_LITTLE)):
    """
    creates a key_point dictionary writes it to json and returns it
    Args:
        dir_path - the function searches for pictures recursivly in it
        file_name - the name of the exported file, also reads the file
        for previous info
        (allowing to stop the function and get back to it later)
        to_print - set that represents the printing options
    Note:
        in the files specified by
        util.JSON_SUPER_BACKUP_FILE and util.JSON_BACKUP_FILE
        backups are being made, so they can be excessed in case of corrupt file
        in file_name
    """
    kp = util.import_json(file_name, to_print)
    count = 0
    back_count = 0
    super_count = 0
    if DIRS not in kp.keys():
        kp[DIRS] = []
    for (cur, dirs, files) in os.walk(DIR_PATH):
        keys = set(kp.keys())
        if cur not in kp[DIRS]:
            for file in files:
                if file not in util.BAD:
                    try:
                        if file not in keys:
                            kp[str(file)] = compare_images(TRAIN_IMAGE,
                                                           cur + "\\" + file)

                            count, back_count, super_count = util.save(
                                count, back_count, super_count,
                                kp, file_name, to_print)
                        if kp[file] is None:
                            print (str(file) + "is None")
                    except ValueError:
                        print "value error in dir:" + cur
                        print file
                        break
            kp[DIRS].append(cur)
        print("finished dir " + str(cur) + " length " + str(len(kp)))
    return kp


def calculate_distances(kp):
    centers = util.import_json(util.JSON_CENTERS)
    distances = []
    for key in kp:
        if not key == DIRS:
            if key in centers:
                mini = float("inf")
                for point in kp[key]:
                    cur_dist = util.point_dist(kp[key][0], centers[key])
                    mini = cur_dist if cur_dist < mini else mini
                distances.append(mini)
            else:
                print(str(key) + " not in centers")
    return distances

def main():
    kp = create_file(DIR_PATH, KP_BF_JASON,
                     to_print=set(util.PRINT_ALL))
    distances = calculate_distances(kp)
    print("Mean value is:" + str(sum(distances)/len(distances)))
    print("Var value is:" + str(np.var(np.array(distances))))
    plt.hist(distances,bins = np.linspace(0,730,74))
    plt.title("Distances Histogram")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    for name, dist in zip(kp,distances):
        if dist > 600:
            PIL.Image.open(util.YES_SMALL_TEST+"\\"+name).show()
    
if __name__ == "__main__":
    main()
