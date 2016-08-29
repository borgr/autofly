# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:49:17 2015

@author: borgr
"""

#import drone_commands
import cv2
import PIL.Image
import time

def wait_webcam(mirror=False):
    """ gets if should make mirror image or not
        writes to IM_PATH the pictures"""
    cam = cv2.VideoCapture(0)
    i = 0
    while True:
        i += 1
        ret_cal, img = cam.read()
        if (img>210).all():
            return True
        if(not ret_cal):
            print ("not")
        print(img[0][0][0])

wait_webcam()
drone_commands.initCopter(r'~/camNapi_test_log')
drone_commands.arm()
print("armed! hooray guy!")
drone_commands.disarm()
print("safe zone guys, cheers")
