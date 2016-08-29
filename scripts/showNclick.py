# -*- coding: utf-8 -*-
"""
shows pictures from given directory (recursivly) and save points on them
"""
#%%
import json
import cv2
import numpy as np
import os
import PIL.Image
import toolbox
from util import *
ix,iy = -1,-1
clicked = []
directory_path = YES_TRAIN

EXIT_MARK = -2
NO_TARGET_MARK = -3
NO_CLICK = -1
assert(EXIT_MARK != NO_TARGET_MARK)
assert(EXIT_MARK != NO_CLICK)
assert(NO_TARGET_MARK != NO_CLICK)


def clickOnMosaic(path, file):
    """
    shows an image to capture a left-click on it.

    input:
        n - no image
        y - write the left click on the picture
        p - like y and print the file-name
        x - save&exit
        """
    mosaic = PIL.Image.open(path + '\\' + file)
    global ix, iy, clicked
    ix, iy = NO_CLICK, NO_CLICK
    clicked = []

    def markImageFromClick(event, x, y, glags, param):
        global ix, iy, clicked
        ix, iy = x, y
        clicked.append((x, y))
    cv2.namedWindow(file, flags = cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(file, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.setMouseCallback(file, markImageFromClick)
    img = np.asarray(mosaic)
    cv2.imshow(file, img)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord("x"): #   ord closes the window
            ix,iy = EXIT_MARK, EXIT_MARK
            break

        if k == ord("n"): #   ord closes the window and says it is has no xy
            ix,iy = NO_TARGET_MARK,NO_TARGET_MARK
            break
        if k == ord("y"):
            break
        if k == ord("p"):
            print (path + '\\' + file)
            break
    cv2.destroyAllWindows()


def main():
    JSON_FILE = r"C:\Users\User\Google Drive\autofly-510\simulator\tmp.txt"
    with open(JSON_FILE,'r') as js:
        centers = json.load(js)
    for key in centers.keys():
        centers[str(key)] = centers.pop(key)

    counter = 0
    visited = set()
    for (cur,dirs,files) in os.walk(directory_path):

        print (cur)
        count = 0
        back_count = 0
        super_count = 0
        ix="d"
        for ufile in files:
            file = str(ufile)
            if file in visited:
                counter += 1
                print(file+" already tagged")
            visited.add(file)
            if file in centers.keys():
                if centers[file] == [NO_TARGET_MARK,NO_TARGET_MARK]:
                    os.rename(cur + '\\' + file,MOVE_NO_TARGET + '\\' + file)
                    centers.pop(file)
                    print(str(file)+"moved")
                elif centers[file] == [NO_CLICK,NO_CLICK]:
                    centers.pop(file)
                    print (str(file) + " no click")
            elif file not in BAD:
                try:
                    if file not in centers.keys():
                        ix,iy = NO_CLICK,NO_CLICK
                        clickOnMosaic(cur, file)
                        if ix == EXIT_MARK:
                            util.export(centers,JSON_FILE)
                            break
                        centers[file] = (ix,iy)
                        count += 1
                        back_count += 1
                        super_count +=1
                        if count > SAVE_RATE:
                            count = 0
                            util.export(centers,JSON_FILE)
                        if back_count > BACKUP_RATE:
                            print"backup"
                            back_count = 0
                            util.export(centers,JSON_BACKUP_FILE)
                        if super_count > SUPER_BACKUP:
                            print"super"
                            super_count = 0
                            util.export(centers,JSON_SUPER_BACKUP_FILE)
                except ValueError:
                    print "value error in dir:" + cur
                    print file
                    break
    #            except:
    #                print "unknown error in dir:" + cur
    #                print file
    #                break

        if (ix == EXIT_MARK):
            break
    print (str(len(centers))+" pictures tagged")
    util.export(centers, JSON_FILE)
    print (str(counter) + " duplicate files")
    if (ix != EXIT_MARK):
        print("no more tagges needed")
if __name__ == "__main__":
    main()