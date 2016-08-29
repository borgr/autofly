# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 13:30:16 2016

@author: User
"""

import sys
import os
sys.path.append(os.path.dirname(__file__) + r"/../database/scripts") ## TODO when in the plane should be replaced and just be in the same dir
sys.path.append(os.path.dirname(__file__) + r"/../odroid_code")
import drone_simulator_commands as dr
#import drone_commands as dr
import learn_existence as ex
import learn_location
import simulator
import Camera
import toolbox
import numpy as np
import cv2
import matplotlib.pyplot as plt
PRINT = False
SIMULATING = True
IMAGE_CENTER = (320, 240)
ALT_GOAL = 1  # goal of altitude for the copter
LOG_PATH = "logFileSimulator"

def init_autofly():
    global LANDING
    global save_locations
    global save_note
    LANDING = False
    save_locations = []
    save_note = None

def prepare():
    init_autofly()
    dr.initCopter(LOG_PATH, mpstate)
    dr.arm(mpstate)
    dr.takeoff(5, mpstate)
    return True


def spiralHeading():
    """ yields spiral destinations, starting from current position"""
    counter = 0
    curRadius = dr.getLocation(mpstate).z ## TODO configure while testing ##
    radiusJumps = dr.getLocation(mpstate).z
    subjumpsNum = 2
    corner = 1
    while True:
        if corner == 1 and counter == 0:
            for i in range(subjumpsNum):
                yield toolbox.Location(0.5 * curRadius / subjumpsNum,
                                       -0.5 * curRadius / subjumpsNum, 0)
        elif corner == 2:
            for i in range(subjumpsNum):
                yield toolbox.Location(0, curRadius / subjumpsNum, 0)
        elif corner == 3:
            for i in range(subjumpsNum):
                yield toolbox.Location(-curRadius / subjumpsNum, 0, 0)
        elif corner == 4:
            for i in range(subjumpsNum):
                yield toolbox.Location(0, curRadius / subjumpsNum, 0)
        elif corner == 5:
            for i in range(subjumpsNum):
                yield toolbox.Location((curRadius + radiusJumps) / subjumpsNum,
                                       -radiusJumps / subjumpsNum, 0)
        corner += 1
        if corner == 5:
            counter += 1
            corner = 1
            curRadius += radiusJumps
            if counter > 3:
                yield -1


def maximumStep():
    """ chooses the maximum step size"""
    print("maximum step does not rely on alt yet")
    return 0.5

if SIMULATING:
    cam = None
    def takeImage():
        """takes Image in the simulator"""
        global save_locations
        loc = dr.getLocation(mpstate)
        z, y, x = loc.z, loc.y, loc.x
        if not save_locations or save_locations[-1] !=(x, y, z):
            save_locations.append((x, y, z))
        return np.asarray(simulator.getPictureFromCamera(
                          Camera.getWebCam(loc), simulator.GRAY_WORLD))
else:
    cam = cv2.VideoCapture(0)
    def takeImage():
        """takes Image in the simulator"""
        if not cam.isOpened():
            dr.logAction("video capture did not start")
            return None
        successful_reading, frame = cap.read()
        if not successful_reading:
            dr.logAction("error while taking picture")
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def isTargetHere():
    counter = 1 if ex.exists(takeImage()) else 0
    # double check target exists
    while counter > 0:
        dr.logAction("target found " + str(counter) + " times")
        counter = counter + 1 if ex.exists(takeImage()) else 0
        if counter == 2:
            dr.logAction("target found enough times")
            return True
        if counter == 0:  ## TODO only for debugging purposes!!
            dr.logAction("target lost")
    return False


def searchForTarget():
    """ searches around until the target is found in the camera,
        returns if found or not"""
    found = isTargetHere()
    heading = spiralHeading()
    target = heading.next()
    while target != -1 and not found:
        dr.fly_to(target.x, target.y, target.z, mpstate)
        found = isTargetHere()
        target = heading.next()
    if target == -1:
        global LANDING
        LANDING = True
        dr.land(mpstate)
    return found


def chooseDecrease():
    """ choose how much lower to go if the target is seen"""
    return 1  # maybe choose not as a constant? by alt


def chooseHeading(centerPoint):
    """ gets a location of the target in pixels
        and chooses relative heading in x,y,z"""
    if PRINT:
        print("choose heading")
    location = dr.getLocation(mpstate)
    cam = Camera.getWebCam(location)
    x = cam.pixelsToMeters(centerPoint[0] - IMAGE_CENTER[0])
    y = cam.pixelsToMeters(centerPoint[1] - IMAGE_CENTER[1])
    z = -chooseDecrease()
    if PRINT:
        print("changing:" + str((x, y, z)))
    return x, y, z


def locate():
    """ locates the target and flies to it returns true if succeded"""
    centerPoint = learn_location.locate(takeImage())
    if centerPoint:
        x, y, z = chooseHeading(centerPoint)
        if PRINT:
            print("closing in")
        dr.fly_to(x, y, z, mpstate)
        return True
    dr.logAction("target was expected but not found")
    return False


def getCloser():
    return locate()


def shouldLand():
    """ a function that decides whether to land or not"""
    return LANDING


def flyToTarget():
    if prepare():
        while(dr.getLocation(mpstate).z > ALT_GOAL) and not shouldLand():
            if searchForTarget():
                getCloser()
                if dr.getLocation(mpstate).z < ALT_GOAL:
#                    searchForTarget()
                    dr.logAction("Target touched")
    else:
        dr.logAction("failed in preparations")
    return dr.getLocation(mpstate)


if SIMULATING:
    def waitForSignal():
        """waits for a signal to start looking"""
        global save_note
        save_note = raw_input("write a note for this simulation")
else:
    def waitForSignal():
        """waits for a signal to start looking"""
        dr.logAction("\nunimplemented wait for signal\n")


def logResults():
    with open('simulatedCourses.txt', 'a') as log_file:
        from drone_simulator_commands import NOISE_FACTOR
        log_file.write("noise factor:" + str(NOISE_FACTOR)+"\n")
        log_file.write(str(save_note)+"\n")
        log_file.write(str(save_locations)+"\n")

def run():
    import time
    dr.prepare()
    time.sleep(3)
    dr.land()

def finish():
    dr.logAction("manually disarming, use close instead")
    while (dr.isarmed(mpstate)):
        dr.disarm(mpstate)
    if cam:
        cam.release()

def logFly():
    ans = flyToTarget()
#    print(save_locations)
    if save_locations:
        if toolbox.point_dist((save_locations[-1][0], save_locations[-1][1]), (0, 0)) < 1:
            logResults()
            print("logged")
    else:
        print("empty locations","-----------------")
#        raise
    return ans

def calculate_success(trials, log=True):
    import time
    global ALT_GOAL
    ALT_GOAL = 1
    x = np.linspace(0.01, 3, 20)
    y_dist = []
    y_success = []
    for noise in x:
        t = time.time()
        dr.NOISE_FACTOR = noise
        if log:
            locations = [logFly() for i in range(trials)]
        else:
            locations = [flyToTarget() for i in range(trials)]
        distances = [toolbox.point_dist((location.x, location.y), (0, 0)) for location in locations]
        if noise > 0.1 and noise < 0.3:
            plt.hist(distances)
            plt.xlabel("distances")
            plt.ylabel("num in bin")
            plt.title("histogram of distances from target (square loss)")
            plt.show()
            plt.scatter([location.x for location in locations], [location.y for location in locations])
            plt.xlabel("x in meters")
            plt.ylabel("y in meters")
            plt.title("finishing locations with normal noise")
            plt.show()
            plt.hexbin([location.x for location in locations], [location.y for location in locations])
            plt.xlabel("x in meters")
            plt.ylabel("y in meters")
            plt.title("finishing locations with normal noise")
            plt.show()
        y_dist.append(np.asarray(distances).mean())
        y_success.append(len([dist for dist in distances if dist < 8**0.5])*100/trials)
    plt.plot(x, y_dist)
    plt.xlabel("noise standard deviation")
    plt.ylabel("meters")
    plt.title("avarage finishing distance by noise")
    plt.show()
    plt.xlabel("noise standard deviation")
    plt.ylabel("percentage")
    plt.title("success rate by noise")
    plt.plot(x, y_success)
    plt.show()
    plt.plot(x[:10], y_dist[:10])
    plt.xlabel("noise standard deviation")
    plt.ylabel("meters")
    plt.title("avarage finishing distance by noise")
    plt.show()
    plt.xlabel("noise standard deviation")
    plt.ylabel("percentage")
    plt.title("success rate by noise")
    plt.plot(x[:10], y_success[:10])
    plt.show()
    elapsed = time.time() - t
    print("time elapsed it", elapsed)


def main():
    import time
    waitForSignal()
    t = time.time()
    flyToTarget()
    if not SIMULATING:
        time.sleep(3)
    dr.land(mpstate)
    finish()
    elapsed = time.time() - t
    print("time elapsed it", elapsed)
    logResults()

if __name__ == "__main__":
    mpstate = None
#    main()
#    calculate_success(30)
    import cProfile
    cProfile.run("simulator.runDemo([(-1.2312851863851713, 0.16351445525023434, 6.3289601446727843), (-0.69103756119618132, 0.26202186615295842, 6.7169465962139885), (0.1422527346791731, 0.39053454218946138, 5.7475254884397646), (1.4644685420599379, -0.94017990662649698, 5.5163086062443725), (3.3641516014585844, -2.6049990717018199, 5.9013602849216822), (4.774609457652689, 0.41366367188935493, 5.3396823565001306), (3.903465843277159, 2.276446356515347, 5.086653459564463), (0.75558403201853608, -0.031520494042253766, 4.5255239225323383), (-1.3759288240771048, -0.13119528627923469, 2.8082258325499465), (0.31105415439402928, 0.65317961211685427, 1.3601595317241091), (0.92290147440996972, 0.2110855553245764, 1.2283337462522208), (2.3059023488290089, -0.59601835070695675, 0.74225539803454466), (1.4257673632578545, 0.052440473270122356, 1.3266524810775373), (0.35352820217086878, 0.035202076847230868, 2.2972258157934577)])")

