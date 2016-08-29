# -*- coding: utf-8 -*-
"""
learning where the center of the target is
Created on Thu Dec 03 12:36:48 2015

@author: borgr
"""
import toolbox
import matplotlib.pyplot as plt
import cv2
import os
import PIL.Image
import numpy as np
THRESHOLD = 400
TRAIN_IMAGE = r'C:\Users\User\Google Drive\autofly-510\database\scripts\20151181442_0003465.jpg'
DIR_PATH = toolbox.YES_SMALL_TEST
#COMPARE_IMAGE = r'C:\Users\User\Google Drive\autofly-510\database\scripts\20151181442_0003655.jpg'
JSON_FILE = '.\\JsonCenters'
KP_BF_JASON = r'.\kp_find_one.json'
DIRS = "dir"  # key of the finished directories list

def locate(img_query):
    return compare_images(TRAIN_IMAGE, img_query)


def compare_images(path1, img_query):
    """
    path1 - train image
    compares 2 images and returns the best matcher"""
    img_test = cv2.imread(path1, 0)
    if img_query is None or img_test is None:
        if img_query is None:
            print("compare_images got an empty image")
        else:
            print("compare_images got an empty test image")
        return None
    surf = cv2.SURF(THRESHOLD)

    #  create features
    kp_test, des_test = surf.detectAndCompute(img_test, None)
    kp_query, des_query = surf.detectAndCompute(img_query, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(crossCheck=True)
    # Match descriptors.
    if des_query is not None and len(des_query):
        matches = bf.match(des_test[1:2], des_query)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        return kp_query[matches[0].queryIdx].pt
    return None


def create_file(dir_path, file_name, to_print=set(toolbox.PRINT_LITTLE)):
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
        toolbox.JSON_SUPER_BACKUP_FILE and toolbox.JSON_BACKUP_FILE
        backups are being made, so they can be excessed in case of corrupt file
        in file_name
    """
    kp = toolbox.import_json(file_name, to_print)
    count = 0
    back_count = 0
    super_count = 0
    if DIRS not in kp.keys():
        kp[DIRS] = []
    for (cur, dirs, files) in os.walk(DIR_PATH):
        keys = set(kp.keys())
        if cur not in kp[DIRS]:
            for file in files:
                if file not in toolbox.BAD:
                    try:
                        if file not in keys:
                            img_query = cv2.imread(cur + "\\" + file, 0)
                            kp[str(file)] = compare_images(TRAIN_IMAGE, img_query)
                            count, back_count, super_count = toolbox.save(
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
    centers = toolbox.import_json(toolbox.JSON_CENTERS)
    distances = []
    for key in kp:
        if not key == DIRS:
            if key in centers:
                distances.append(toolbox.point_dist(
                                            kp[key], centers[key]))
            else:
                print(str(key) + " not in centers")
    return distances

def main():
    kp = create_file(DIR_PATH, KP_BF_JASON,
                     to_print=set(toolbox.PRINT_LITTLE))
    distances = calculate_distances(kp)
    print("Mean value is:" + str(sum(distances)/len(distances)))
    print("Var value is:" + str(np.var(np.array(distances))))
    plt.hist(distances,bins = np.linspace(0,730,74))
    plt.title("Distances Histogram")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    for name, dist in zip(kp,distances):
        if dist > 600:
            PIL.Image.open(toolbox.YES_SMALL_TEST+"\\"+name).show()
    
    #
    ##draw matches
    #for match in matches:
    #        print(match.queryIdx)
    #        img2 = cv2.drawKeypoints(img_test,[kp_test[match.queryIdx]],None,(255,0,0),4)
    #        plt.imshow(img2),plt.show()
    #        img2 = cv2.drawKeypoints(img_query,[kp_query[match.trainIdx]],None,(255,0,0),4)
    #        plt.imshow(img2),plt.show()
    
    
    
    ###compare best k
    #img2 = cv2.drawKeypoints(img_query,kp_query,None,(255,0,0),4)
    #plt.imshow(img2),plt.show()
    #bf = cv2.BFMatcher()
    #
    #matches = bf.knnMatch(des_test[0:2],des_query, k=2)
    #print matches
    ## Apply ratio test
    #good = []
    #for m,n in matches:
    #    if m.distance < 0.75*n.distance:
    #        good.append([m])
    #
    ##img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
    #print good
    ##plt.imshow(img3),plt.show()
    
    #print keypoint numbers and pictures
    #for i in range(len(kp)):
    #    if(kp[i].pt[0]>250 and kp[i].pt[0]<450 and kp[i].pt[1]>100 and kp[i].pt[1]<200):    
    #        print(i)
    #        img2 = cv2.drawKeypoints(img,[kp[i]],None,(255,0,0),4)
    #        plt.imshow(img2),plt.show()
    #print des
    #print kp
if __name__ == "__main__":
    main()
