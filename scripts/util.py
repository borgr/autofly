# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 08:44:57 2015

@author: borgr
"""
import json
import os
import math

BAD = {'desktop.ini', 'Thumbs.db'}
YES_FULL = "..\\yes"
YES_SMALL_TEST = YES_FULL + "\\6yes"  #  1489 pics
YES_SMALL_TRAIN = YES_FULL + "\\someparts"
NO_FULL = "..\\no"
NO_SMALL_TEST = NO_FULL + "\\6no"  #  1541 pics
YES_SMALL_TRAIN = NO_FULL + "part1"
YES_COMBINE = YES_FULL + "\\combine"
YES_TRAIN = YES_FULL

JSON_CENTERS = '.\\JsonCenters'
JSON_BACKUP_FILE = '.\\JasonBackup'
JSON_SUPER_BACKUP_FILE = '.\\Super_JasonBackup'
MOVE_NO_TARGET = '..\\\yesToNo'
SAVE_RATE = 5
BACKUP_RATE = 16
SUPER_BACKUP = 100

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

