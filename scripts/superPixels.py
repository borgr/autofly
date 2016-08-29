# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 15:10:48 2015

@author: borgr
"""

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

TRAIN_IMAGE = r'C:\Users\borgr\Google Drive\autofly-510\database\yes\yes5\20151181442_0003465.jpg'
TRAIN_CENTER = [153,368]
#TRAIN_IMAGE = r'C:\Users\borgr\Google Drive\autofly-510\database\yes\part2\2015118152_0000018.jpg'
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(TRAIN_IMAGE))


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
    to_check = [tuple(pixel_place)]
    super_pixel_nums = set([super_num])
    pixels = [tuple(pixel_place)]  # list of pixels to return
    checked = set()
    # look for all pixels in neighbor superpixels
    while to_check:
        current = to_check.pop()
        checked.add(tuple(current))
        neighbbors = [(current[0]+1,current[1]),(current[0]+1,current[1]+1),(current[0]+1,current[1]-1),(current[0]-1,current[1]),(current[0]-1,current[1]+1),(current[0]-1,current[1]-1),(current[0],current[1]-1),(current[0],current[1]+1)]
        for neighbor in neighbbors:
            # do not check twice and make sanity check
            if (neighbor not in checked) and (in_bounds(neighbor, super_img)):
#                for current in pixels:
#                    if super_img[current[0],current[1]] not in [100, 101, 102, 114, 115, 87]:
#                            print("error"+str(current))
#                            raise
                # if you are a neighbor of a pixel in the original super pixel 
                # you are a neighbor for sure
                if super_img[current[0], current[1]] == super_num:
                    pixels.append(neighbor)
                    to_check.append(neighbor)
                    # if you are out of the original superpixel 
                    # then all of this superpixel  should be added
                    if super_img[neighbor[0], neighbor[1]] not in super_pixel_nums:
                        if super_img[current[0], current[1]] != 42 or len (super_pixel_nums)>2:
                            pass
                        super_pixel_nums.add(
                                        super_img[neighbor[0], neighbor[1]])
                else:
                    if super_img[current[0], current[1]] == super_img[neighbor[0], neighbor[1]]:
                        pixels.append(neighbor)
                        to_check.append(neighbor)
#    for current in pixels:
#        if super_img[current[0],current[1]] not in [100, 101, 102, 114, 115, 87]:
#                print("error"+str(current))                        
    return pixels

 
# loop over the number of segments
for numSegments in [200]:
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments = numSegments, sigma = 5)
    neighbors = find_neighbors(TRAIN_CENTER, image, segments)
    for neighbor in neighbors:
#        if segments[neighbor[0],neighbor[1]] == segments[TRAIN_CENTER[0],TRAIN_CENTER[1]]:
            image[neighbor[0],neighbor[1]] = 0
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
     
# show the plots
plt.show()
#print(find_neighbors((479,400), image, segments))