# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 08:44:57 2015

@author: borgr
"""
import json
import os
#import mpl_toolkits.basemap.pyproj as pyproj

BAD = {'desktop.ini', 'Thumbs.db'}
YES_FULL = "..\\yes"
YES_SMALL_TEST = YES_FULL + "\\6yes"  #  1489 pics
YES_SMALL_TRAIN = YES_FULL + "\\someparts"
NO_FULL = "..\\no"
NO_SMALL_TEST = NO_FULL + "\\6no"  #  1541 pics
YES_SMALL_TRAIN = NO_FULL + "part1"
YES_COMBINE = YES_FULL + "\\combine"

JSON_CENTERS = '.\\JsonCenters'
JSON_BACKUP_FILE = '.\\JasonBackup'
JSON_SUPER_BACKUP_FILE = '.\\Super_JasonBackup'
MOVE_NO_TARGET = '..\\\yesToNo'
SAVE_RATE = 51
BACKUP_RATE = 203
SUPER_BACKUP = 503

#  printing options
PRINT_ERRORS = "errors"
PRINT_ALL = "all"
PRINT_LITTLE = "little"


def save(count, back_count, super_count, data, file_name, to_print):
    count += 1
    back_count += 1
    super_count += 1
    if count > SAVE_RATE:
        count = 0
        export(data, file_name, to_print)
    if back_count > BACKUP_RATE:
        back_count = 0
        export(data, JSON_BACKUP_FILE, to_print)
    if super_count > SUPER_BACKUP:
        if PRINT_LITTLE in to_print or PRINT_ALL in to_print:
            print ("gone through another " +
                   str(SUPER_BACKUP) +
                   " pictures")
        super_count = 0
        export(data, JSON_SUPER_BACKUP_FILE, to_print)
    return count, back_count, super_count

def import_json(path, to_print=set()):
    """
    imports json file from path returns an empty dict if does not exist
    """
    if not os.path.exists(path):
        if to_print:
            print(path + " does not exist")
        return {}
    with open(path, 'r') as js:
        centers = json.load(js)
    for key in centers.keys():
        centers[str(key)] = centers.pop(key)
    if to_print:
        print(" imported sucssesfully with length " + str(len(centers)))
    return centers


def point_dist((x1, y1), (x2, y2)):
    """euclidian distance between two points"""
    return ((1.0*x1-x2)**2+(1.0*y1-y2)**2)**0.5


#class Location:
#    def __init__(self, lat, lon, alt):
#        self.lat = lat
#        self.lon = lon
#        self.alt = alt
#
#def addXYZtoLocation(location, x, y, z):
#    ##TODO make pyproj work!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#    old_x, old_y, old_z = location.lon, location.lat, location.alt
#    wgs84 = pyproj.Proj("+init=EPSG:4326")
#    isn2004 = pyproj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1")
#    old_x, old_y, old_z = pyproj.transform(wgs84, isn2004, location.lon, location.lat, location.alt)
#    new_x = old_x + x
#    new_y = old_y + y
#    new_z = old_z + z
#    ##TODO make pyproj work
##    print("addXYZtoLocation unimplemented")
##    new_lon, new_lat, new_alt =new_x, new_y, new_z
#    new_lon, new_lat, new_alt =  pyproj.transform(isn2004, wgs84, new_x, new_y, new_z)
#    return Location(new_lat, new_lon, new_alt)

class Location:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def addXYZtoLocation(location, x, y, z):
    return Location(location.x + x, location.y + y, location.z + z)


def export(obj, file_name, backup=JSON_BACKUP_FILE,
           to_print=set([PRINT_ERRORS])):
    try:
        with open(file_name, 'w') as js:
            json.dump(obj, js)
            if PRINT_ALL in to_print:
                print(str(file_name + " dumped succesfully"))
    except ValueError:
        with open(backup, 'r') as back:
            with open(file_name, 'w') as js:
                js.write(back.read())
        if PRINT_ERRORS in to_print:
            print ("unicode error while writing")

        