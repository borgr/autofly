# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 13:51:56 2016

@author: borgr
"""

import toolbox
import numpy as np
NOISE_FACTOR = 0.02
logFile = None
vehicle = None
location = None
armed = False


def initCopter(logFilePath, mpstate):
    """ initializing the copter. CALL THIS BEFORE FLIGHT"""
    global vehicle
    global logFile
    global location  # x, y, z
    logFile = logFilePath
    noise = np.random.normal(0, 3, 2)
    location = toolbox.Location(-abs(noise[0]), -abs(noise[1]), 0)


def logAction(message):
    """ logs given massage in a logFile, for later use.
    """
#    outFile = open(logFile, 'a')
#    outFile.write(message + "\n")
#    outFile.close()
#    print(message)


def isarmed(mpstate):
    return armed


def arm(mpstate):
    logAction("arm() called")
    logAction('ARMED!')
    global armed
    armed = True


def disarm(mpstate):
    logAction("disarm() called")
    logAction('DISARMED')
    global armed
    armed = False


def takeoff(aTargetAltitude, mpstate):
    logAction('Taking off!')
    return fly_to(0, 0, aTargetAltitude, mpstate)


def fly_to(x, y, z, mpstate):
    """ flies x meters to the right, y meters forward, and z meters up """
    noise = np.random.normal(0, max(z/4, NOISE_FACTOR), 3)
    global location
    logAction("trying to fly to: (" + str(x) + "," +
              str(y) + "," + str(z) + ")")
    location = toolbox.addXYZtoLocation(getLocation(mpstate),
                                        x + noise[0],
                                        y + noise[1],
                                        z + noise[2])
#    print ("location is" + str((location.x, location.y, location.z)))
    return (location.x, location.y, location.z)


def land(mpstate):
    """lands on the ground on the place it is now"""
    ##TODO doesn't do ANYTHING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("landing unimplementd")
    return fly_to(0, 0, 0, mpstate)


def getLocation(mpstate):
    """ returns a struct containing the location (x, y, z) """
    return location


