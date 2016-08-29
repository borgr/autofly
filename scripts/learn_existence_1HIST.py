# -*- coding: utf-8 -*-
"""
learning if a target is in or is not in a picture
Created on Thu Dec 03 12:36:08 2015

@author: borgr
"""
from skimage.color import rgb2gray
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import exposure
import matplotlib.pyplot as plt

import PIL.Image
import cv2
import os
import numpy as np
import toolbox
from sklearn import metrics


HIST_COMPARE_METHOD = cv2.cv.CV_COMP_CHISQR
THRESHOLD = 400
SEG_NUM = 200
TRAIN_IMAGE = r'..\..\database\yes\yes5\20151181442_0003465.jpg'
TRAIN_CENTER = (153,368)
DIR_PATH_WITHOUT = toolbox.NO_SMALL_TEST
DIR_PATH_WITH = toolbox.YES_SMALL_TEST
COMPARE_IMAGE = r'..\..\database\no\6no\20151181442_0007557.jpg'
JSON_FILE = '.\\JsonCenters'
TRIAL = 1
KP_BF_JASON_WITHOUT = r'.\kp_exist_1hist_no_' + str(TRIAL) + r'.json'
KP_BF_JASON_WITH = r'.\kp_exist_1hist_yes_' + str(TRIAL) + r'.json'
DIRS = "dir"  # key of the finished directories list


def read_image(file):
    return io.imread(file)
    # return cv2.imread(cur + "\\" + file, 0)


def preprocess_image(img):
    return get_superpixels(img), img_as_float(equalize_hist(convert2gray(img)))


def convert2gray(img):
    return rgb2gray(img)
    # return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def calculate_hist(img):
#    return np.bincount(img, None, 256) 
    return cv2.calcHist([img], [0], None, [256], [0,1])


def equalize_hist(img):
    return exposure.equalize_hist(img)
    # return cv2.equalizeHist(img1)


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
    return:
    a list of pixels that are in the ame superpixel as the pixel_place
    or in neighboring superpixels
    """
    super_num = super_img[pixel_place[0], pixel_place[1]]
    to_check = [tuple(pixel_place)]
    super_pixel_nums = set([super_num])
    pixels = set([tuple(pixel_place)])  # list of pixels to return
    checked = set()
    # look for all pixels in neighbor superpixels
    while to_check:
        current = to_check.pop()
        checked.add(tuple(current))
        neighbbors = [(current[0]+1,current[1]),(current[0]+1,current[1]+1),(current[0]+1,current[1]-1),(current[0]-1,current[1]),(current[0]-1,current[1]+1),(current[0]-1,current[1]-1),(current[0],current[1]-1),(current[0],current[1]+1)]
        for neighbor in neighbbors:
            # do not check twice and make sanity check
            if (neighbor not in checked) and (in_bounds(neighbor, super_img)):
                # if you are a neighbor of a pixel
                # in the original super pixel
                # you are a neighbor for sure
                if super_img[current[0], current[1]] == super_num:
                    pixels.add(tuple(neighbor))
                    to_check.append(tuple(neighbor))
                    # if you are out of the original superpixel
                    # then all of this superpixel  should be added
                    if (super_img[neighbor[0], neighbor[1]]
                            not in super_pixel_nums):
                        super_pixel_nums.add(
                                        super_img[neighbor[0], neighbor[1]])
                # if you arin the same superpixel as the current
                # one you are a neighbor
                else:
                    if (super_img[current[0], current[1]] ==
                            super_img[neighbor[0], neighbor[1]]):
                        pixels.add(tuple(neighbor))
                        to_check.append(tuple(neighbor))
    pixels = list(pixels)
    return pixels


def get_superpixels(ndarray):
    """gets an image(ndarray) and returns the array of superpixels(segments)
    """
    # load the image and convert it to a floating point data type
    image = img_as_float(ndarray)

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments=SEG_NUM,
                    sigma=5, enforce_connectivity=True)

#    # show results
#    fig = plt.figure("Superpixels -- %d segments" % (SEG_NUM))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.imshow(mark_boundaries(image, segments))
#    plt.axis("off")
#    plt.show()
    return segments


def index2D(l, elem):
    """ finds the first row and column elem exists in l where l is 2Darray"""
    for row, i in enumerate(l):
        for column, val in enumerate(i):
            if elem == val:
                return row, column
    print ("element " + str(elem) + " was not found")
    return -1, -1


def compare_images(img_test, img_query, test_center, test_hist):
    """
    path1 - train image
    compares 2 images and returns the best matcher
    returns:
        (xPixel,yPixel),matcher_distance
        if no pixel found (-1,-1),float("inf")
        """
    query_segments, img_query = preprocess_image(img_query)
    best_dist = float("inf")
    # check distance for each segment from the test
    for seg_num in range(max((x for y in query_segments for x in y))+1):
        pixel = index2D(query_segments, seg_num)
        if pixel[0] != -1:
            compare_hist = get_super_hist(pixel, img_query, query_segments)
            dist = compare_hists(test_hist, compare_hist)
#            print(seg_num)
#            if (seg_num == query_segments[test_center[0], test_center[1]]):
##                print("test_hist" + str([x-y for x in test_hist[0] for y in get_super_hist(pixel, 
##                                                img_query, query_segments)[1][0]]))
##                print("hist"+str(get_super_hist(pixel, 
##                                                img_query, query_segments)[1]))
#                print("distance" + str(dist))
            if dist < best_dist:
                best_pixel = pixel
                best_dist = dist

#    print("best pixel found is" + str(best_pixel))
#    print("best_dist found is" + str(best_dist))
#    show_region(img_query, [] , query_segments)
#    show_region(img_query, get_super_pixels(query_segments, query_segments[best_pixel[0], best_pixel[1]]), query_segments)
    
    return best_pixel, best_dist

def get_super_pixels(segments, seg):
    """ returns iterable of pixel locations of the segment"""
    colors = []
    for row, i in enumerate(segments):
        for column, val in enumerate(i):
            if seg == val:
                colors.append((row,column))
    return colors


def show_hist(hist, colors):
    plt.scatter([range(len(hist))],hist)
    plt.hist(colors, len(colors))
    plt.show()


def get_super_hist(pixel, img, segments):
    """
    returns the histogram of the superpixel neighbors of the given pixel.
    ARGS:
    pixel - (x,y) of target pixel
    img
    segments
    returns:
    (segments of img, histogram)
    """
    img = convert2gray(img)
    seg_num = segments[pixel[0], pixel[1]]

    colors = np.asarray([img[pix[0], pix[1]] for pix in 
              get_super_pixels(segments, seg_num)])
    colors = colors.astype(np.float32)
    hist = calculate_hist(colors)

    return hist


def show_region(img, region, segments, color=0):
    # show selected region
    for pixel in region:
        img[pixel[0], pixel[1]] = color
    fig = plt.figure("Superpixels -- %d segments" % (SEG_NUM))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img, segments))
    plt.axis("off")
    plt.show()


def compare_hists(hist1, hist2):
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)
    return cv2.compareHist(hist1, hist2, HIST_COMPARE_METHOD)


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
    segments1, img1 = preprocess_image(read_image(TRAIN_IMAGE))
    test_hist = get_super_hist(TRAIN_CENTER, img1, segments1)
    print("training pic processing done")
    if DIRS not in kp.keys():
        kp[DIRS] = []
    for (cur, dirs, files) in os.walk(dir_path):
        keys = set(kp.keys())
        if cur not in kp[DIRS]:

            for file in files:
                if file not in toolbox.BAD:
                    try:
                        if file not in keys:
                            img2 = read_image(cur + "\\" + file)
                            kp[str(file)] = compare_images(img1, img2,
                                                           TRAIN_CENTER, 
                                                           test_hist)
                            count, back_count, super_count = toolbox.save(
                                count, back_count, super_count,
                                kp, file_name, to_print)
                        if kp[file] is None:
                            print (str(file) + "is None")
                    except ValueError:
                        print ("value error in dir:" + cur)
                        print (file)
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
        print ("print_statistics got None")
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
    print("running unitests")
#    comparing with compare img
    segments, equalized1 = preprocess_image(read_image(TRAIN_IMAGE))
    
#    img1 = cv2.imread(TRAIN_IMAGE, 0)
#    print("not usingequalization")
    hist = get_super_hist(TRAIN_CENTER, equalized1, segments)
    assert(compare_hists(hist, hist) == 0)
    tmp_segments = get_superpixels(read_image(TRAIN_IMAGE))
    assert((segments == tmp_segments).all())
    seg_num = segments[TRAIN_CENTER[0], TRAIN_CENTER[1]]
    pixel = index2D(segments, seg_num)
    try:
        pix_neigh = find_neighbors(pixel, equalized1, segments)
        center_neigh = find_neighbors(TRAIN_CENTER, equalized1, segments)
        assert([x for x in pix_neigh if x not in center_neigh] == [])
        assert([x for x in center_neigh if x not in pix_neigh] == [])
    except AssertionError as e:
        pix_neigh = find_neighbors(pixel, equalized1, segments)
        center_neigh = find_neighbors(TRAIN_CENTER, equalized1, segments)
        show_region(equalized1, pix_neigh, segments, 250)
        show_region(equalized1, center_neigh, segments, 0)
        print([x for x in pix_neigh if x not in center_neigh])
        print([x for x in center_neigh if x not in pix_neigh])
        raise e
    compare_hist = get_super_hist(pixel, equalized1, segments)
    assert(compare_hists(hist, compare_hist) == 0)
    assert(segments[pixel[0], pixel[1]] == seg_num)
    tmp_hist = get_super_hist(pixel, equalized1, segments)
    not_in_both = [amount for (i, amount) in enumerate(hist[0]) if amount != tmp_hist[1][i]]
    assert(not_in_both == [])
    tmp_segments, equalized1_copy = preprocess_image(read_image(TRAIN_IMAGE))
    assert((segments == tmp_segments).all())
#    print("comparing with itself")
    pixel, dist = compare_images(equalized1, read_image(TRAIN_IMAGE), TRAIN_CENTER, hist)
    assert(dist == 0)
    print("unitests passed")
#    print("comparing with compare image")
#    img2 = read_image(COMPARE_IMAGE)
#    print(compare_images(equalized1, img2, TRAIN_CENTER, hist))
#    compare_images(TRAIN_IMAGE, COMPARE_IMAGE,TRAIN_CENTER, hist)
#    raise
    print("starting to calculate")
    y = []
    scores = []
    
    # test compare_images on yes
    kp_yes = create_file(DIR_PATH_WITH, KP_BF_JASON_WITH, to_print=set(toolbox.PRINT_LITTLE))
    print_statistics(kp_yes)
    
    # calculate roc
    for val in get_dist_iter(kp_yes):
        if val is not None:
            y.append(-1)
            scores.append(val)
    print("calculating pictures without target")
    
    #  test compare_images on no
    kp = create_file(DIR_PATH_WITHOUT, KP_BF_JASON_WITHOUT, to_print=set(toolbox.PRINT_LITTLE))
    print_statistics(kp)

    # calculate roc
    for val in get_dist_iter(kp):
        if val is not None:
            y.append(1)
            scores.append(val)
    
    print("calculating pictures with target")

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
