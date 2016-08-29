# -*- coding: utf-8 -*-
"""
learning if a target is in or is not in a picture
Created on Thu Dec 03 12:36:08 2015

@author: borgr
"""
import cv2
import os
import util
import matplotlib.pyplot as plt
from sklearn import metrics
from skimage.segmentation import slic
from skimage.util import img_as_float

THRESHOLD = 400
SEG_NUM = 150
TRAIN_IMAGE = r'C:\Users\borgr\Google Drive\autofly-510\database\yes\yes5\20151181442_0003465.jpg'
DIR_PATH_WITHOUT = util.NO_SMALL_TEST
DIR_PATH_WITH = util.YES_SMALL_TEST
COMPARE_IMAGE = r'C:\Users\borgr\Google Drive\autofly-510\database\no\6no\20151181442_0007557.jpg'
JSON_FILE = '.\\JsonCenters'
TRIAL = 1
KP_BF_JASON_WITHOUT = r'.\kp_exist_no_' + str(TRIAL) + r'.json'
KP_BF_JASON_WITH = r'.\kp_exist_yes_' + str(TRIAL) + r'.json'
DIRS = "dir"  # key of the finished directories list


def in_bounds(pixel, img):
    """
    checks if a pixel(x,y) is in bounds of an img(ndarray)
    """
    if (pixel[0] < 0 or pixel[1] < 0 or
       pixel[0] >= img.shape[0] or pixel[1] >= img.shape[1]):
        return False
    return True


def find_neighbors(pixel_place, img, super_img):
    """finds and returns the list of neighbor pixels
    ARGS:
    pixel_place - (x,y) of a pixel to get neighbors of
    img - image
    super_img - ndarray with clasification to super_pixels
    """
    super_num = super_img[pixel_place[0], pixel_place[1]]
    to_check = [pixel_place]
    super_pixel_nums = [super_num]
    pixels = [pixel_place]  # list of pixels to return
    checked = set()
    while to_check:
        current = to_check.pop()
        checked.add(current)
        neighbbors = [(current[0]+1,current[1]),(current[0]+1,current[1]+1),(current[0]+1,current[1]-1),(current[0]-1,current[1]),(current[0]-1,current[1]+1),(current[0]-1,current[1]-1),(current[0],current[1]-1),(current[0],current[1]+1)]
        for neighbor in neighbbors:
            if neighbor not in checked and in_bounds(neighbor, super_img):
                pixels.append(neighbor)
                if neighbor[0] == 480:
                    print("")
                if super_img[current[0]][current[1]] == super_num:
                    to_check.append(neighbor)
                    if super_img[neighbor[0], neighbor[1]] != super_num:
                        super_pixel_nums.append(
                                        super_img[neighbor[0], neighbor[1]])
                else:
                    if super_img[neighbor[0], neighbor[1]] in super_pixel_nums:
                        to_check.append(neighbor)
    return pixels


def get_superpixels(ndarray):
    """gets an image(ndarray) and returns the array of superpixels
    """
    # load the image and convert it to a floating point data type
    image = img_as_float(ndarray)

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments=SEG_NUM, sigma=5)
    return segments


def compare_images(img_test, img_query):
    """
    path1 - train image
    compares 2 images and returns the best matcher
    returns:
        (xPixel,yPixel),matcher_distance
        """

    surf = cv2.SURF(THRESHOLD)

    #  create features
    kp_test, des_test = surf.detectAndCompute(img_test, None)
    kp_query, des_query = surf.detectAndCompute(img_query, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(crossCheck=True)

    # Match descriptors.
    if des_query is not None:
        matches = bf.match(des_test[1:2], des_query)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

#        #draw matches
#        for match in matches:
#            print(match.queryIdx)
#            img2 = cv2.drawKeypoints(img_test,[kp_test[match.queryIdx]],None,(255,0,0),4)
#            plt.imshow(img2),plt.show()
#            img2 = cv2.drawKeypoints(img_query,[kp_query[match.trainIdx]],None,(255,0,0),4)
#            plt.imshow(img2),plt.show()
#            print match.distance

        return kp_query[matches[0].queryIdx].pt, matches[0].distance
    return None, None


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
    img1 = cv2.imread(TRAIN_IMAGE, 0)
    if DIRS not in kp.keys():
        kp[DIRS] = []
    for (cur, dirs, files) in os.walk(dir_path):
        keys = set(kp.keys())
        if cur not in kp[DIRS]:
            for file in files:
                if file not in util.BAD:
                    try:
                        if file not in keys:
                            img2 = cv2.imread(cur + "\\" + file, 0)
                            kp[str(file)] = compare_images(img1, img2)
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
        print("finished dir " + str(cur) + " total length " + str(len(kp)))
    return kp


def get_dist_iter(kp):
    """
    takes kp dictionary and creates an iterator over the pictures
    Note: it is a dictionary hence order is not guarenteed
    """
    for key, val in kp.iteritems():
        if key != DIRS:
            yield val[1]


def print_statistics(kp):
    if kp is None:
        print "print_statistics got None"
        return
    else:
        summ = 0.0
        mini = float("inf")
        for val in get_dist_iter(kp):
            if val is not None:
                if mini > val:
                    mini = val
                summ += val
    print ("mean " + str(summ/(len(kp)-1)))
    print("minimum " + str(mini))


def main():
    #    compare_images(TRAIN_IMAGE, COMPARE_IMAGE)
    y = []
    scores = []

    #  test compare_images on no
    kp = create_file(DIR_PATH_WITHOUT, KP_BF_JASON_WITHOUT, to_print=set(util.PRINT_LITTLE))
    print_statistics(kp)

    # calculate roc
    for val in get_dist_iter(kp):
        if val is not None:
            y.append(-1)
            scores.append(val)
    # test compare_images on yes
    kp_yes = create_file(DIR_PATH_WITH, KP_BF_JASON_WITH, to_print=set(util.PRINT_LITTLE))
    print_statistics(kp_yes)

    # calculate roc
    for val in get_dist_iter(kp_yes):
        if val is not None:
            y.append(1)
            scores.append(val)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    print(thresholds[2])

    # plot roc
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve of learn existence')
    plt.legend(loc="lower right")
    plt.show()
    
#    for i in np.linspace(0.3,0.5,7):
#        count_yes = 0
#        count_no = 0
#        threshold = i
#        for key, val in kp.iteritems():
#            if key != DIRS:
#                if val[1] < threshold:
#                    count_no += 1
#        for key, val in kp_yes.iteritems():
#            if key != DIRS:
#                if val[1] < threshold:
#                    count_yes += 1
#        print("threshold: " + str(threshold))
#        print("yes " + str(count_yes))
#        print("no " + str(count_no))

#            print ("min " + str(min(kp, lambda x: x[1])))
#            print ("mean " + str(sum(kp, lambda x: x[1])/len(kp)))




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
