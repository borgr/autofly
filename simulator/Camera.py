# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:51:28 2015

@author: guy
"""
import math
import numpy as np
from drone_simulator_commands import NOISE_FACTOR
# parameters for the c910 sensor webCam we got
FOV_X = 70.58 * math.pi / 180
FOV_Y = 55.92 * math.pi / 180
NUM_X_PIXELS = 640
NUM_Y_PIXELS = 480
DRONE_PIXEL_LOCATION = (320, 240)


#class Camera():
#    """ hold parameters of camera """
#    def __init__(self, alttitude, longtitude, latitude, fovX, fovY,
#                 numPixelsX, numPixelsY, pitch=0, roll=0):
#        self.alttitude = alttitude    # THATS THE Z
#        self.longtitude = longtitude  # THATS THE Y
#        self.latitude = latitude      # THATS THE X
#        self.fovX = fovX
#        self.fovY = fovY
#        self.numPixelsX = numPixelsX
#        self.numPixelsY = numPixelsY
#        self.pitch = pitch
#        self.roll = roll
#        # calculating the size in meters for half on the image, then
#        # dividing by the number of pixels so we get the Meter / pixel ratio
#        self.numMeterInPixel = (alttitude * math.tan(0.5 * fovX)) / (0.5 *
#                                numPixelsX)
#
#
#def getWebCam(x, y, z):
#    """ returns a new c910 sensor webCam in the given (x,y,z) location """
#    return Camera(z, y, x, FOV_X, FOV_Y, NUM_X_PIXELS, NUM_Y_PIXELS)


class Camera():
    """ hold parameters of camera """
    def __init__(self, z, y, x, fovX, fovY,
                 numPixelsX, numPixelsY, pitch=0, roll=0):
        self.z = z    # THATS THE Z
        self.y = y  # THATS THE Y
        self.x = x      # THATS THE X
        self.fovX = fovX
        self.fovY = fovY
        self.numPixelsX = numPixelsX
        self.numPixelsY = numPixelsY
        self.pitch = pitch
        self.roll = roll
        # calculating the size in meters for half on the image, then
        # dividing by the number of pixels so we get the Meter / pixel ratio
        self.numMeterInPixel = (z * math.tan(0.5 * fovX)) / (0.5 *
                                                             numPixelsX)

    def pixelsToMeters(self, pixels):
        return pixels * self.numMeterInPixel


def getWebCam(loc):
    """ returns a new c910 sensor webCam in the given (x,y,z) location """
    noise = np.random.normal(0,NOISE_FACTOR,3)
    return Camera(loc.z + noise[2], loc.y + noise[1], loc.x + noise[0], FOV_X, FOV_Y, NUM_X_PIXELS, NUM_Y_PIXELS)
