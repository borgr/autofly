# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:03:27 2015

@author: guy
"""
#import drone_commands
import Camera
import math
import mpl_toolkits.basemap.pyproj as pyproj
import cv2

NUM_X_PIXELS = 640
NUM_Y_PIXELS = 480

MIN_HEIGHT_FROM_TARGET = 0.8
MIN_DIST_TO_EDGE = math.tan(0.5 * Camera.FOV_X) * MIN_HEIGHT_FROM_TARGET
CLOSING_ON_TARGET_DISTANCE = 0.1

class Location:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
def addXYZtoLocation(location, x, y, z):
    wgs84=pyproj.Proj("+init=EPSG:4326")
    isn2004=pyproj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1")
    old_x, old_y, old_z = pyproj.transform(wgs84, isn2004, location.lon, location.lat, location.alt)
    new_x = old_x + x
    new_y = old_y + y
    new_z = old_z + z
    new_lon, new_lat, new_alt =  pyproj.transform(isn2004, wgs84, new_x, new_y, new_z)
    return Location(new_lat, new_lon, new_alt)
        

def distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    dist = ( (x2 - x1)**2 + (y2 - y1)**2 ) ** 0.5
    return dist

def distanceFromEdgeOfImage(imageSize, pixelLoc, metersToPixelRatio):
    x, y = 0, 1
    distanceX = metersToPixelRatio * min(imageSize[x] - pixelLoc[x],
                                         pixelLoc[x])
    distanceY = metersToPixelRatio * min(imageSize[y] - pixelLoc[y],
                                         pixelLoc[y])
    return min(distanceX, distanceY)




def navigateToTargetLocation(curImage, targetLocation):
    loc = drone_commands.getLocation()
    loc = Location(loc.lat, loc.lon, loc.alt)
    camera = Camera.getWebCam(loc.lat, loc.lon, loc.alt)
    distToEdge = distanceFromEdgeOfImage(
                 curImage.shape, targetLocation, camera.numMeterInPixel)
    if distToEdge < MIN_DIST_TO_EDGE:
        return Location(loc.lat, loc.lon, loc.alt - CLOSING_ON_TARGET_DISTANCE)
        #drone_commands.fly_to(loc.lat, loc.lon, loc.alt - CLOSING_ON_TARGET_DISTANCE)
    else:
        oldPixel = drone_commands.DRONE_PIXEL_LOCATION
        x_diff = targetLocation[0] - oldPixel[0]
        y_diff = targetLocation[1] - oldPixel[1]
        metersToPixels = camera.numMeterInPixel
        newLoc = addXYZtoLocation(loc, x_diff * metersToPixels, y_diff * metersToPixels, 0)
        return Location(newLoc.lat, newLoc.lon, newLoc.alt)
        #drone_commands.fly_to(newLoc.lat, newLoc.lon, newLoc.alt)
    
def searchTarget(startLocation, metersToPixels):
    radiusJumps = 100
    curRadius = 100
    corner = 1
    while True:
        if corner == 1:
            yield addXYZtoLocation(startLocation, curRadius * metersToPixels,0,0)
        elif corner == 2:
            yield addXYZtoLocation(startLocation, 0, curRadius * metersToPixels, 0)
        elif corner == 3:
            yield addXYZtoLocation(startLocation, -curRadius * metersToPixels, 0, 0)
        elif corner == 4:
            yield addXYZtoLocation(startLocation, 0, -curRadius * metersToPixels, 0)
        corner += 1
        if corner == 5:
            corner = 1
            curRadius += radiusJumps
        
        
        