import PIL.Image
import cv2
import numpy as np
import math
import Camera
import sys
import os
sys.path.append(os.path.dirname(__file__) + r"/../database/scripts") ## TODO when in the plane should be replaced and just be in the same dir
sys.path.append(os.path.dirname(__file__) + r"/../odroid_code")
sys.path.append(os.path.dirname(__file__) + r"/3D")

import toolbox
import perspective
#import cv2

class World():
    """ hold parameters given world that the simulator fly in """
    def __init__(self, image, numPixelsInMeter, targetLocation, perspectiveImage, copterImage, logo):
        self.image = PIL.Image.open(image)
        self.logo = PIL.Image.open(logo)
        self.copterImage = PIL.Image.open(copterImage)
        self.perspectiveImage = PIL.Image.open(perspectiveImage)
        self.numPixelsInMeter = numPixelsInMeter
        self.targetLocation = targetLocation

    def metersToPixels(self, x):
        """ converts a real world meters to amount of pixels in the
            simulated world"""
        return self.numPixelsInMeter * x

def euclidianDist(point1, point2):
    """ calculates the euclidian distance between 2 points """
    x1,y1 = point1
    x2,y2 = point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

# parameters for the gray world picture
gray_world_numpixels_for_meter = 314.1391300570479
#gray_world_target_location = (2427, 1688)
#GRAY_WORLD = World(r"gray_world.JPG", gray_world_numpixels_for_meter, gray_world_target_location)

#gray_world_target_location = (2536, 1715)
#GRAY_WORLD =  World(r"double_gray_world.png", gray_world_numpixels_for_meter, gray_world_target_location)
gray_world_numpixels_for_meter = 128.2076054009458
gray_world_target_location = (3102, 2682)
GRAY_WORLD =  World(r"panorama_world.tif", gray_world_numpixels_for_meter, gray_world_target_location, r"all pics\_DVD3991.TIF", r"3D\copterCropped.bmp", r"3D\autofly.png")
def find_area(camera, world):
    # using triangular similarity in order to calculate which part of the image
    # to cut. u can see drawing of the geometry at autofly-510\simulator\getPictureFromCamera_geomtry.pptx

    # finds the indexes in the world image that contains the information the camera will see
    cx = float(camera.numPixelsX)
    cz = (0.5*cx) / math.tan(0.5 * camera.fovX)
    z = world.metersToPixels(float(camera.z))
    x = (cx * z) / cz
    x_target, y_target = world.targetLocation
    left = int(round(world.metersToPixels(float(camera.x)) - (x/2) + x_target))
    right = int(round(world.metersToPixels(float(camera.x)) + (x/2) + x_target))
    cy = float(camera.numPixelsY)
    # this cz and the last cz should be pretty much equal, if input is correct
    cz = (0.5*cy) / math.tan(0.5 * camera.fovY)
    z = world.metersToPixels(float(camera.z))
    y = (cy * z) / cz
    lower = int(round(world.metersToPixels(float(camera.y)) - (y/2) + y_target))
    upper = int(round(world.metersToPixels(float(camera.y)) + (y/2) + y_target))
    return left, lower, right, upper


def getPictureFromCamera(camera, world):
    """ gets camera and world, and returns the picture that the camera
        see if it was in the given world - image returned as PIL.Image format """
    left, lower, right, upper = find_area(camera, world)
    #cropping the image and resizing it so it will match how to camera see it
    res = world.image.crop((left,lower,right,upper))
    res = res.resize((camera.numPixelsX, camera.numPixelsY))
    return res


def getDemoPoints():
    points = [(5,i,5) for i in range(5,0,-1)] + [(i,0,5) for i in range(5,0,-1)] + [(0,0,i) for i in range(5,1,-1)] + [(0,0,0.8),(0,0,0.5),(0,0,0.3),(0,0,0.1)]
    newPoints = []
    for pointIndex in range(len(points) - 1):
        newPoints.append(points[pointIndex])
        newPoints.append(tuple([(points[pointIndex][i] + points[pointIndex+1][i])/2.0 for i in range(3)]))
    return newPoints


cur_index = 0
def show(img):
    global cur_index
    cur_index += 1
    """show an image"""
#    #cv2
#    import cv2
#    print("showing pic")
#    name = "fly"
#    cv2.namedWindow(name, flags = cv2.WINDOW_NORMAL)
#    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
#    img = np.asarray(img)
#    cv2.imshow(name, img)
#    cv2.destroyAllWindows()
#    NO_CLICK = ""
#    NO_TARGET_MARK = -1
#    global ix, iy, clicked
#    ix, iy = NO_CLICK, NO_CLICK
    clicked = []
    name = "Autofly"
    sleepTime = 1 # choose timebetween pictures in seconds

#    def markImageFromClick(event, x, y, glags, param):
#        global ix, iy, clicked
#        ix, iy = x, y
#        clicked.append((x, y))
#    cv2.destroyAllWindows()
    cv2.namedWindow(name, flags = cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
#    cv2.setMouseCallback(file, markImageFromClick)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(r"C:\Users\owner\Desktop\flights\less_noisy_jpg\img" + str(cur_index) + ".jpg",
                img)
    cv2.imshow(name, img)
    k = True
    while k:
        k = cv2.waitKey(10*sleepTime) & 0xFF
        k = False
#    import matplotlib.pyplot as plt
#    #only works for wx backaend
#    mng = plt.get_current_fig_manager()
#    mng.frame.Maximize(True)
#
#    #only works for Qt backend
#    figManager = plt.get_current_fig_manager()
#    figManager.window.showMaximized()
#    img.show()
#
#    #only works for Tk backend
#    from matplotlib import pyplot as plt
#    mng = plt.get_current_fig_manager()
#    mng.full_screen_toggle()
#    plt.show()

def getColorArea(camera, world):
    """ this function returns the world colored in the seen area"""
    left, lower, right, upper = find_area(camera, world)
    img = world.image.copy()
    obj = img.load()
    for i in range(left,right):
        for j in range(lower, upper):
            try:
                obj[i,j] = (100 + obj[i,j][0], obj[i,j][1],obj[i,j][2])
            except:
                pass
#            print(i,j)
#    print(left, lower, right, upper)
#    import time
#    time.sleep(1)
#    raise
#    print(img.shape)
#    print(np.array(img).shape)
#    print("----------------")
    return np.array(img)

def combined_imgs(x,y,z):
    img1 = getPictureFromCamera(Camera.getWebCam(toolbox.Location(x, y, z)), GRAY_WORLD)
    img2 = getColorArea(Camera.getWebCam(toolbox.Location(x, y, z)), GRAY_WORLD)
    img1 = cv2.resize(np.array(img1), (np.array(img2).shape[1],np.array(img2).shape[0]))
    img2 = np.array(img2)
    img1 = np.array(img1)
    img12 = np.concatenate((img1, img2), axis=1)

#    other half
    img3 = perspective.putCopterOnImage(GRAY_WORLD.perspectiveImage, GRAY_WORLD.copterImage, (x,y,z))
    img3 = np.array(img3)
    img3 = cv2.resize(np.array(img3), (int(np.array(img12).shape[1]/2), np.array(img12).shape[0]))
#    imgBlack = np.zeros(img3.shape)
    logo = cv2.resize(np.array(GRAY_WORLD.logo), (int(np.array(img12).shape[1]/2), np.array(img12).shape[0]))
    img3 = np.concatenate((logo, img3), axis=1)
    img3 = np.array(img3)
    img12 = np.concatenate((img3, img12), axis=0)
    return img12


def runDemo(points, path=None, name=None):
    subpointsNum = 1
    subpointsNum += 2
    lastPoint = None
    if path is None:
        print("showing path")
        for curPoint in points:
            if lastPoint:
                for x, y, z in zip(np.linspace(lastPoint[0], curPoint[0], subpointsNum),
                                   np.linspace(lastPoint[1], curPoint[1], subpointsNum),
                                   np.linspace(lastPoint[2], curPoint[2], subpointsNum)):
                    show(combined_imgs(x,y,z))
            lastPoint = curPoint
        k = True
        while k:
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        print("saving path to file")
        i = 0
        for point in points:
            i += 1
            getPictureFromCamera(Camera.getWebCam(toolbox.Location(point[0],point[1],point[2])), GRAY_WORLD).save(path + '//' + name + str(i) + ".tif")
def main():
    runDemo([(-6.646701732023427, -1.8622162439366499, 5.623553489511476), (-6.5772718930181773, -1.4228223259760109, 6.7535277809464631), (-5.3793397777398884, -3.8051258983577601, 5.356589298980138), (-3.4438565504629555, -5.6140011181472405, 4.5177212564259444), (-5.1620128906394136, -2.8596292179546365, 5.1885404615284418), (-3.5076847188105389, -0.81346236035048225, 5.8735239968882604), (1.8790414434863298, 0.069902331771523052, 3.9415749579068704), (0.85735713980326489, 1.4162147485263652, 2.5814444460523429), (1.3283256571167632, 0.5266173155644196, 2.2287867081990416), (0.43312980058707939, 0.30357335720574846, 2.1741749926034002), (0.03312980058707939, 0.00357335720574846, 1.1741749926034002)])
#    runDemo([(1,0,3),(1,0,2),(1,1,2), (0,1,2), (0,2,2)])
#    import cProfile
#    cProfile.run("runDemo([(-1.9459717972196042, -6.4456014970656241, 5.559784620446278), (0.60187094480650005, -10.530956707630324, 3.8015578506108598), (-1.4591289758799362, -13.821149966182023, 3.9515855602455203), (-1.408683555039441, -12.05729440080421, 1.7404599291987712), (-2.0697394642359472, -7.0491139748156204, 1.3914075294790933), (-3.3280991394868771, -5.4942232991913436, 1.2449519176919477), (-9.4229103246090791, -4.5824692985001896, 5.8316467979629687), (-9.1694036111098995, -3.382504657794819, 9.0615684909875576), (-6.3779835128163622, -2.0799265833558578, 10.682137816138226), (0.38429550560606884, 2.5518538557360952, 11.548816581725143), (2.4680832586577575, -0.66871763181043198, 12.111606796318611), (-2.5547485290822012, -0.24645576122665958, 7.2672212153144891), (0.080088416760380809, -0.75263086198407103, 3.6369702957323549)])")

if __name__ == "__main__":
    mpstate = None
    main()