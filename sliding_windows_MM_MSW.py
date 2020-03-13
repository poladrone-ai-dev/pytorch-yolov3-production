from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

from PIL import Image
from skimage.io import imread
from pyimagesearch.helpers import sliding_window
from skimage import io
import os
import sys
import time
import datetime
import argparse
import json
import numpy as np
import cv2
import torch
import shutil
import threading
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

###############################################################################
#
#
#
###############################################################################
import math
import tifffile as tiff

from termcolor import colored
from texttable import Texttable
import csv
import osgeo.ogr, osgeo.osr
from osgeo import ogr
from osgeo import gdal
from os import system, name
import geocoder


###############################################################################
#
#
#
###############################################################################
t_Table = []
t_Table.append(['         Description                ', '          Elapsed Time ( Sec)              '])


###############################################################################
#
#
#
###############################################################################
def clear():
    if name == 'nt':
        _ = system('cls')

    else:
        _ = system('clear')

    ###############################################################################


#
#
#
###############################################################################
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print(colored('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), 'cyan', attrs=['bold']), end=printEnd)
    if iteration == total:
        print()


###############################################################################
#
#
#
###############################################################################
def reverseGeocode(coordinates):
    result = rg.search(coordinates)
    # pprint.pprint(result)
    return result


###############################################################################
#
#
#
###############################################################################
DatumEqRad = [6378137.0,
              6378137.0,
              6378137.0,
              6378135.0,
              6378160.0,
              6378245.0,
              6378206.4,
              6378388.0,
              6378388.0,
              6378249.1,
              6378206.4,
              6377563.4,
              6377397.2,
              6377276.3];
DatumFlat = [298.2572236,
             298.2572236,
             298.2572215,
             298.2597208,
             298.2497323,
             298.2997381,
             294.9786982,
             296.9993621,
             296.9993621,
             293.4660167,
             294.9786982,
             299.3247788,
             299.1527052,
             300.8021499];

Item = 0  # // default
a = DatumEqRad[Item]  # // equatorial radius (meters)
f = 1 / DatumFlat[Item]  # // polar flattening
drad = math.pi / 180  # // convert degrees to radians

# // Mor constants, extracted from the function:
k0 = 0.9996  # // scale on central meridian
b = a * (1 - f)  # // polar axis
e = math.sqrt(1 - (b / a) * (b / a))  # // eccentricity
e0 = e / math.sqrt(1 - e * e)  # // called e' in reference
esq = (1 - (b / a) * (b / a))  # // e² for use in expansions
e0sq = e * e / (1 - e * e)


def utmToLatLon(x, y, utmz, north):
    if x < 160000 and x > 840000:
        print("Outside permissible range of easting values.")
        return

    if (y < 0):
        print("Negative values are not allowed for northing.")
        return

    if y > 10000000:
        print("Northing may not exceed 10,000,000.")
        return;

    zcm = 3 + 6 * (utmz - 1) - 180;
    e1 = (1 - math.sqrt(1 - e * e)) / (1 + math.sqrt(1 - e * e))
    M0 = 0

    if (north):
        M = M0 + y / k0
    else:
        M = M0 + (y - 10000000) / k0

    mu = M / (a * (1 - esq * (1 / 4 + esq * (3 / 64 + 5 * esq / 256))))
    phi1 = mu + e1 * (3 / 2 - 27 * e1 * e1 / 32) * math.sin(2 * mu) + e1 * e1 * (
                21 / 16 - 55 * e1 * e1 / 32) * math.sin(4 * mu)
    phi1 = phi1 + e1 * e1 * e1 * (math.sin(6 * mu) * 151 / 96 + e1 * math.sin(8 * mu) * 1097 / 512)
    C1 = e0sq * math.pow(math.cos(phi1), 2)
    T1 = math.pow(math.tan(phi1), 2)
    N1 = a / math.sqrt(1 - math.pow(e * math.sin(phi1), 2))
    R1 = N1 * (1 - e * e) / (1 - math.pow(e * math.sin(phi1), 2))
    D = (x - 500000) / (N1 * k0)
    phi = (D * D) * (1 / 2 - D * D * (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * e0sq) / 24)
    phi = phi + math.pow(D, 6) * (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * e0sq - 3 * C1 * C1) / 720
    phi = phi1 - (N1 * math.tan(phi1) / R1) * phi

    outLat = math.floor(1000000 * phi / drad) / 1000000

    lng = D * (1 + D * D * ((-1 - 2 * T1 - C1) / 6 + D * D * (
                5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * e0sq + 24 * T1 * T1) / 120)) / math.cos(phi1)
    lngd = zcm + lng / drad

    outLon = math.floor(1000000 * lngd) / 1000000

    return [outLat, outLon]


###############################################################################
# Calculates the intersection over union between two boxes
#
# Parameters:
# boxA (dict): box 
###############################################################################
def calculate_iou(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA["x2"] - boxA["x1"] + 1) * (boxA["y2"] - boxA["y1"] + 1)
    boxBArea = (boxB["x2"] - boxB["x1"] + 1) * (boxB["y2"] - boxB["y1"] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou, interArea, boxAArea, boxBArea


###############################################################################
# Window detection with tensorflow model
#
# Parameters:
# opt_window_size (int): size of detection window
# opt_conf_thres (float): confidence threshold to register a positive detection
# opt_nms_thres (float): non maximal suppression threshold
# window (PIL.Image): Pillow image representation of window
# sess (tf.Session): Tensorflow session to run inference with

# Returns:
# output_tensor (tf.Tensor): Tensor representation of window detection
###############################################################################
def detect_image_tensorflow(opt_window_size, opt_conf_thres, opt_nms_thres, window, sess=None):
    nms_thres = opt_nms_thres
    conf_thres = opt_conf_thres
    rows = opt_window_size
    cols = opt_window_size

    # inp = window
    inp = np.asarray(window)
    inp = inp[:, :, [2, 1, 0]]
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    num_detections = int(out[0][0])
    detection_output = []

    # print("Number of Total Detections out from the model =>>> %d" % num_detections)
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > conf_thres:
            # print("Class ID = >>%d" % classId + "  score  =>>> %f" % score)
            x1 = bbox[1] * cols
            y1 = bbox[0] * rows
            x2 = bbox[3] * cols
            y2 = bbox[2] * rows

            detection_output.append([x1, y1, x2, y2, score, score, classId - 1])

    output_tensor = torch.FloatTensor(detection_output)
    return output_tensor


###############################################################################
#  Image detection function with PyTorch models. Applies resize, padding, and
#  to Tensor transformations, before running detection and NMS
#
# Paramenters:
# opt_window_size (int): size of the detection window
# opt_conf_thres (float): confidence threshold for registering a positive detection
# opt_nms_thres (float): non maximal suppression threshold
# window (PIL.Image): PIL image representation of detection window
# model (torch.nn): PyTorch model for detection
    
# Returns:
# detections[0] (torch.Tensor): tensor representation of window detection
###############################################################################
def detect_image(opt_window_size, opt_conf_thres, opt_nms_thres, window, model):
    img_size = 416  # don't change this, because the model is trained on 416 x 416 images
    conf_thres = opt_conf_thres
    nms_thres = opt_nms_thres
    window_height, window_width = window.size
    # scale and pad image
    ratio = min(img_size / window_width, img_size / window_height)
    imw = round(window_width * ratio)
    imh = round(window_height * ratio)

    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), \
                                                         max(int((imw - imh) / 2), 0)), (128, 128, 128)),
                                         transforms.ToTensor(), ])

    # convert PIL image to Tensor
    # img = Image.fromarray(window, 'RGB')
    img = window
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    return detections[0]

###############################################################################
# Sorts the json values in ascending value, with x1 having higher priority, followed by y1 having priority
#
# Parameters:
# opt_Debug (bool): specifies whether to run this function in debug mode
# opt_output (string): output path to save the sorted detections
# detections (dict): bbox detections in json format

# Returns:
# sorted_detections_list (dict): detection dict sorted in ascending order
###############################################################################
def SortDetections(opt_Debug, opt_output, detections):
    detections_list = []

    for box in detections:
        detections_list.append({box: detections[box]})

    sorted_detections = sorted(detections_list, key=lambda item: (list(item.values())[0]['x1'], list(item.values())[0]['y1']))
    sorted_detections_dict = {}

    for i in range(len(sorted_detections)):
        key = list(sorted_detections[i].keys())[0]

        for kvp in sorted_detections[i].values():
            sorted_detections_dict[key] = kvp

    if opt_Debug:
        with open(os.path.join(opt_output, "detections_sorted.json"), 'w') as json_fp:
            json.dump(sorted_detections_dict, json_fp, indent=4)
            
    return sorted_detections_dict


###############################################################################
# Converts and exports detection json to a .csv format (Redwan).
# Returns the csv values in the form of GPS fields
#    
# Parameters:
# gen_Csv (bool): whether to generate csv outputs
# detection_json (dict): detection json file in memory
# output_path (str): path where the text file will be exported to
# xOrigin (int): 
# yOrigin (int):
# pixelWidth (int):
# pixelHeight (int):
# rescale (float): 
    
# Returns
# gps_Fields (list): CSV output in GPS format
###############################################################################

def ExportJsonToCSV2(gen_Csv, detection_json, output_path, xOrigin, yOrigin, pixelWidth, pixelHeight, rescale):
    # csv_file = os.path.splitext(os.path.basename(output_path))[0] + '_detection_filtered.csv'
    csv_file = "detections_filtered.csv"
    gps_Fields = []

    for box in detection_json:
        gps_Fields.append([box.replace("box", ""), detection_json[box]["conf"],
                           (-((int(detection_json[box]["center_y"]) * pixelHeight) / rescale) + yOrigin),
                           (((int(detection_json[box]["center_x"]) * pixelWidth) / rescale) + xOrigin), pixelHeight,
                           pixelWidth, yOrigin, xOrigin, rescale])

    if gen_Csv:
        with open(os.path.join(output_path, csv_file), 'w') as detection:
            writer = csv.writer(detection)
            writer.writerow(
                ['ID', 'Score', 'X1', 'Y1', 'Width', 'Height', 'Centroid_X', 'Centroid_Y', 'Latitude', 'Longitude'])
        with open(os.path.join(output_path, csv_file)) as detection:
            lines = detection.readlines()
            last_line = lines[len(lines) - 1]
            lines[len(lines) - 1] = last_line.rstrip()
        with open(os.path.join(output_path, csv_file), 'w') as detection:
            detection.writelines(lines)
        with open(os.path.join(output_path, csv_file), 'a+') as detection:
            for box in detection_json:
                detection.write(box.replace("box", "") + "," + str(detection_json[box]["conf"]) + ","
                                + str(detection_json[box]["x1"]) + "," + str(detection_json[box]["y1"]) + ","
                                + str(detection_json[box]["width"]) + "," + str(detection_json[box]["height"]) + ","
                                + str(detection_json[box]["center_x"]) + "," + str(detection_json[box]["center_y"]) + ","
                                + str(
                    (-((int(detection_json[box]["center_y"]) * pixelHeight) / rescale) + yOrigin)) + ","
                                + str(
                    (((int(detection_json[box]["center_x"]) * pixelWidth) / rescale) + xOrigin)) + "\n")

        detection.close()

    return gps_Fields


###############################################################################
# Converts and exports detectin json to .csv format (Brian)
#
# Parameters:
# detection_json (dict): detection json in memory
# output_path (str): path to export csv file to 
###############################################################################
def ExportJsonToCSV(detection_json, output_path):
    csv_file = os.path.splitext(os.path.basename(output_path))[0] + '_detection_filtered.csv'

    with open(os.path.join(output_path, csv_file), 'w') as detection:
        writer = csv.writer(detection)
        writer.writerow(['ID', 'Score', 'Centroid_X', 'Centroid_Y'])
    with open(os.path.join(output_path, csv_file)) as detection:
        lines = detection.readlines()
        last_line = lines[len(lines) - 1]
        lines[len(lines) - 1] = last_line.rstrip()
    with open(os.path.join(output_path, csv_file), 'w') as detection:
        detection.writelines(lines)

    with open(os.path.join(output_path, csv_file), 'a+') as detection:
        for box in detection_json:
            detection.write(box.replace("box", "") + "," + str(detection_json[box]["conf"]) + ","
                            + str(detection_json[box]["center_x"]) + "," + str(detection_json[box]["center_y"]) + "\n")

    detection.close()


###############################################################################
# Converts and exports detection json to a .txt format
#
# Parameters:
# detection_json (dict): detection json file in memory
# output_path (str): path where the text file will be exported to
###############################################################################
def ExportJsonToText(detection_json, output_path):
    export_Json_to_Text_start = time.time()

    txt_file = os.path.splitext(os.path.basename(output_path))[0] + '_detection_filtered.txt'

    with open(os.path.join(output_path, txt_file), 'a+') as detection:
        for box in detection_json:
            detection.write(box.replace("box", "") + " " + str(detection_json[box]["conf"]) + " " + str(
                detection_json[box]["x1"]) + " "
                            + str(detection_json[box]["y1"]) + " " + str(detection_json[box]["x2"]) + " "
                            + str(detection_json[box]["y2"]) + '\n')

    detection.close()

    export_Json_to_Text_end = time.time()
    t_Table.append(['Export Results from Json to Text ', (export_Json_to_Text_end - export_Json_to_Text_start)])


###############################################################################
#
#
#
###############################################################################
def ConvertFullmap(temp_Folder, tiff_Path, down_scale):
    filename = tiff_Path

    try:
        ds = gdal.Open(filename)
    except:
        print("Could not find/open Tiff map file!!! ... Please enter a valid Tiff file/path!")
        sys.exit()

    prj = ds.GetProjection()
    proj = osgeo.osr.SpatialReference(wkt=ds.GetProjection())
    proj_type = proj.GetAttrValue('AUTHORITY', 1)

    t_Table.append(['TiFF Proj Type ', proj_type])

    if proj.GetAttrValue('AUTHORITY', 1) == None:
        print("Could not find any supported projection for your Tiff file !!! ... Please check your Tiff file!")
        sys.exit()

    center_x = round(ds.RasterXSize / 2)
    center_y = round(ds.RasterYSize / 2)

    xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()
    posX = px_w * center_x + rot1 * center_y + xoffset
    posY = rot2 * center_x + px_h * center_y + yoffset
    posX += px_w / 2.0
    posY += px_h / 2.0
    crs = osgeo.osr.SpatialReference()
    crs.ImportFromWkt(ds.GetProjectionRef())

    crsGeo = osgeo.osr.SpatialReference()
    crsGeo.ImportFromEPSG(int(proj_type))
    t = osgeo.osr.CoordinateTransformation(crs, crsGeo)
    (lat, long, z) = t.TransformPoint(posX, posY)

    srs = osgeo.osr.SpatialReference(wkt=prj)
    N_S = 'false'

    if srs.IsProjected:
        zone = srs.GetAttrValue('projcs')

    if not zone:

        LAT_center = long
        LONG_center = lat

    else:
        zone_1 = (zone[-3:])[0:2]
        zone_2 = (zone[-3:])[2]
        if zone_2 == 'N':
            N_S = 'ture'
        LAT_center, LONG_center = utmToLatLon(lat, long, int(zone_1), N_S)

    coordinates = (LAT_center, LONG_center)

    # cord = reverseGeocode(coordinates)
    cord = geocoder.osm(coordinates, method='reverse')

    t_Table.append(['Map Center Location Lat/Long ', '( ' + str(LAT_center) + ' , ' + str(LONG_center) + ' )'])
    t_Table.append(['Map Location Details ',
                    'City: ' + cord.city + ", " + 'State: ' + cord.state + ", " + 'Country: ' + cord.country])

    rescale = down_scale

    file_name, file_extension = os.path.splitext(filename)

    jpg_file = temp_Folder + "\\" + os.path.splitext(os.path.basename(file_name))[0] + ".jpg"

    try:

        Conv_tiff_jpg_PIL_start = time.time()

        im = tiff.imread(filename)
        im = Image.fromarray(im[:, :, 0:3], 'RGB')

        Conv_tiff_jpg_PIL_end = time.time()

        t_Table.append(['TIFF Reading/Conversion ', (Conv_tiff_jpg_PIL_end - Conv_tiff_jpg_PIL_start)])

    except:

        Conv_tiff_jpg_GDAL_start = time.time()

        PNGDriver = gdal.GetDriverByName("PNG")
        PNGDriver.CreateCopy(jpg_file, ds, 0)

        Conv_tiff_jpg_GDAL_end = time.time()

        t_Table.append(['Convert TIFF to JPG ', (Conv_tiff_jpg_GDAL_end - Conv_tiff_jpg_GDAL_start)])

    return xoffset, yoffset, px_w, -rot2, proj_type, jpg_file, im, cord


###############################################################################
# Creates .shp file from gps coordinates
# 
# Parameters:
# tiff_path (str): temporary path where the TIF file is stored
# gps_Fields (list): list of gps coordinates 
# proj_type (int): coordinate projection type
    
# Precondition: gps_fields are correct, proj_type is an existing projection type
###############################################################################
def ExportShpProj(tiff_path, gps_Fields, proj_type):
    export_shp_proj_start = time.time()

    EPSG_code = proj_type
    shp_file = os.path.splitext(os.path.basename(tiff_path))[0] + '.shp'
    full_shp_path = tiff_path + "\\" +  shp_file
    export_shp = full_shp_path

    spatialReference = osgeo.osr.SpatialReference()
    spatialReference.ImportFromEPSG(int(EPSG_code))
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    shapeData = driver.CreateDataSource(export_shp)
    layer = shapeData.CreateLayer('layer', spatialReference, osgeo.ogr.wkbPoint)
    layer_defn = layer.GetLayerDefn()
    index = 0

    fields_names = ['ID', 'Score', 'Lat', 'Long', 'pixH', 'pixW', 'yO', 'xO', 'Scale']

    for field in fields_names:
        new_field = ogr.FieldDefn(field, ogr.OFTString)
        layer.CreateField(new_field)

    for k in range(len(gps_Fields)):
        point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)

        point.AddPoint(float(gps_Fields[k][3]), float(gps_Fields[k][2]))
        feature = osgeo.ogr.Feature(layer_defn)
        feature.SetGeometry(point)
        feature.SetFID(index)

        for field in fields_names:
            i = feature.GetFieldIndex(field)
            feature.SetField(i, gps_Fields[k][i])

        layer.CreateFeature(feature)
        index += 1

    shapeData.Destroy()

    export_shp_proj_end = time.time()

    t_Table.append(['Export Results to Shp Proj ', (export_shp_proj_end - export_shp_proj_start)])


###############################################################################
# Checks two boxes to see if one is a subset of another
# 
# Parameters:
# boxA (dict): bounding box values for boxA
# boxB (dict): bounding box values for boxB
    
# Returns:
# bool to indicate if one box is within another
###############################################################################
def isSubset(boxA, boxB):

    if boxA["x1"] > boxB["x1"] and boxA["y1"] > boxB["y1"] and boxA["x2"] < boxB["x2"] and boxA["y2"] < boxB["y2"]:
        return True
    elif boxB["x1"] > boxA["x1"] and boxB["y1"] > boxA["y1"] and boxB["x2"] < boxA["x2"] and boxB["y2"] < boxA["y2"]:
        return True

    return False

###############################################################################
# Filters out overlapping bounding boxes, taking only the bounding box with the highest confidence value.
# 
# Parameters:
# opt_Debug (bool): whether to run the function in debug mode
# progress_Counter (int): progress bar counter ranging from 0 to 100
# detections_json (dict): dictionary containing the detections info of every bounding boxes
# iou_thres (float): intersection over union threshold
    
# Returns:
# detections_json (dict): filtered detection json
    
# Precondition: bounding boxes are already sorted in ascending x1 values
###############################################################################
def filter_bounding_boxes_optimized(opt_Debug, progress_Counter, detections_json, iou_thres):
    bounding_boxes_filter_start = time.time()

    deleted_boxes = []
    same_conf_boxes = []
    num_boxes = len(detections_json)
    
    if opt_Debug:
        print("Number of boxes before filtering: " + str(num_boxes))
        
    detections_json_list = list(detections_json)

    progress_percentage_for_filtering = 10
    total_Box_Count = len(detections_json_list)
    

    for idx in range(len(detections_json_list)):
        progress_Counter = progress_Counter + (progress_percentage_for_filtering / total_Box_Count)
        printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

        neighbor_boxes = []
        neighbor_range = 101
        
        for neighbor_idx in range(1, neighbor_range):
            if idx + neighbor_idx < len(detections_json):
                # if isSubset(detections_json[detections_json_list[idx]], detections_json[detections_json_list[idx + neighbor_idx]]):
                    # if detections_json[detections_json_list[idx]]["conf"] < detections_json[detections_json_list[idx + neighbor_idx]]["conf"]:
                    #     neighbor_boxes.append(detections_json_list[idx + neighbor_idx])
                    # else:
                    #     neighbor_boxes.append(detections_json_list[idx])
                
                # else:
                    neighbor_boxes.append(detections_json_list[idx + neighbor_idx])
                 
        for box in neighbor_boxes:
            boxA = detections_json_list[idx]
            boxB = box
            iou, interArea, boxAArea, boxBArea = calculate_iou(detections_json[boxA], detections_json[boxB])

            if iou > iou_thres or isSubset(detections_json[boxA], detections_json[boxB]):
                if detections_json[boxA]["conf"] == detections_json[boxB]["conf"] \
                        and boxA not in same_conf_boxes and boxB not in same_conf_boxes:
                    rand_num = random.randint(1, 2)

                    if rand_num == 1:
                        if boxA not in deleted_boxes and boxA not in same_conf_boxes:
                            deleted_boxes.append(boxA)
                            same_conf_boxes.append(boxA)
                    elif rand_num == 2:
                        if boxB not in deleted_boxes and boxB not in same_conf_boxes:
                            deleted_boxes.append(boxB)
                            same_conf_boxes.append(boxB)

                if detections_json[boxA]["conf"] < detections_json[boxB]["conf"]:
                    if boxA not in deleted_boxes:
                        deleted_boxes.append(boxA)
                elif detections_json[boxB]["conf"] < detections_json[boxA]["conf"]:
                    if boxB not in deleted_boxes:
                        deleted_boxes.append(boxB)

    if opt_Debug:
        print("Number of deleted boxes: " + str(len(deleted_boxes)))

    for box in list(detections_json):
        if box in deleted_boxes:
            del detections_json[box]

    if opt_Debug:
        print("Number of boxes after filtering: " + str(len(detections_json.keys())))

    bounding_boxes_filter_end = time.time()

    # t_Table.append(['Total Number of Trees Found  ', (len(new_detections_json.keys()) + 1)])

    return detections_json


###############################################################################
# Draws bounding boxes on the image with OpenCV, based on coordinates from detection json
#
# Parameters:
# output_json(dict): detection json info for each bounding box
# image (np array): numpy array representation of the image
# output_path (str): path to save the output image
    
# Pre-Cond: bounding boxes coordinates in output_json are within the dimensions of the image
###############################################################################
def draw_bounding_boxes(opt_Debug, output_json, image, output_path, color_dict=None):
    draw_bounding_boxes_start = time.time()
    
    for box in output_json:

        x1 = output_json[box]["x1"]
        y1 = output_json[box]["y1"]
        x2 = output_json[box]["x2"]
        y2 = output_json[box]["y2"]
        width = output_json[box]["width"]
        height = output_json[box]["height"]
        cls_pred = output_json[box]["cls_pred"]
        conf = output_json[box]["conf"]

        if opt_Debug:
            if color_dict is not None:
                color = color_dict[output_json[box]["model"]]
            else:
                color = (255, 0, 0)
        else:
            color = (255, 0, 0)
    
        image = np.asarray(image)
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, box + "-" + str(conf), (int(x1), int(y1)), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            
    # io.imsave(output_path, image)
    image = Image.fromarray(image)
    b,g,r = image.split()
    image = Image.merge("RGB", (r,g,b))
    image.save(output_path)
    draw_bounding_boxes_end = time.time()
    # t_Table.append(['Drawing Results -- Boxes  ', (draw_bounding_boxes_end - draw_bounding_boxes_start) ])

###############################################################################
# draws out center points at tree crowns
#
# Parameters
# output_json(dict): detection json info for each bounding box
# image (np array): numpy array representation of the image
# output_path (str): path to save the output image
###############################################################################
def draw_circles(output_json, image, output_path):
    draw_circles_start = time.time()
    for box in output_json:
        center_x = output_json[box]["center_x"]
        center_y = output_json[box]["center_y"]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 5)

    io.imsave(output_path, image)
    draw_circles_end = time.time()
    # t_Table.append(['Drawing Results -- Circles  ', (draw_circles_end - draw_circles_start) ])

###############################################################################
# Check each sliding windows to see if most of the pixels are black.
# 
# Parameters:
# window (generator): a generator representation of the current sliding window
# winW (int): width of the sliding window
# winW (int): height of the sliding window
    
# Returns:
# (bool): whether most of the pixels in sliding window is black
###############################################################################
def IsBackgroundMostlyBlack(window, winW, winH):
    try:
        w_Gray = Image.fromarray(window).convert('L')
        clrs_Thre = 30
        w_BW = w_Gray.point(lambda x: 0 if x < (0 + clrs_Thre) else 255 if x > (255 - clrs_Thre) else x, '1')
        w_clrs = w_BW.getcolors()

        if (((w_clrs[len(w_clrs) - 1][0]) / (winW * winH)) > (0.70) or ((w_clrs[0][0]) / (winW * winH)) > (0.70)):
            return True

    except Exception as e:
        print(e)

    return False


###############################################################################
# Writes detections (before and after filtering) in json, csv, text, and output image.
#  
# Parameters:
# opt_Debug (bool): whether to run the function in debug mode
# progress_Counter (int): progress count from 0 to 100
# image (np array): numpy array representation of image
# output_path (str): path to write the detection output files
#
# Returns:
# (int) length of the detection json
# input_json(dict): detection json (before filter)
###############################################################################
def GenerateDetections(opt_Debug, progress_Counter, image, output_path, color_dict=None):
    threads = []

    if os.path.isfile(os.path.join(output_path, 'detection.json')):
        with open(os.path.join(output_path, 'detection.json'), 'r') as json_file:
            input_json = json.load(json_file)

        image = np.asarray(image)

        image_before_filter = copy.deepcopy(image)
        before_filter_thread = threading.Thread(target=draw_bounding_boxes,
                                                args=[opt_Debug, input_json, image_before_filter,
                                                        os.path.join(output_path,os.path.basename(output_path) + "_detection_before_filter.jpeg"), color_dict])

        before_filter_thread.start()
        threads.append(before_filter_thread)

        input_json = SortDetections(opt_Debug, output_path, input_json)

        image_width = (image.shape[1])
        image_height = (image.shape[0])

        iou_thres_range = [0.5]
        filtering_start = time.time()

        for iou_thres in iou_thres_range:
            input_json = filter_bounding_boxes_optimized(opt_Debug, progress_Counter, input_json, iou_thres)

        filtering_end = time.time()

        print("Bounding box filtering elapsed time: " + str(filtering_end - filtering_start))

        with open(os.path.join(output_path, "detection_filtered.json"), "w") as img_json:
            json.dump(input_json, img_json, indent=4)

        image_detect = copy.deepcopy(image)
        draw_box_start = time.time()
        box_thread = threading.Thread(target=draw_bounding_boxes, args=[opt_Debug, input_json, image_detect,
                                                                        os.path.join(output_path, os.path.basename(output_path) + "_detection.jpeg"), color_dict])
        box_thread.start()
        threads.append(box_thread)
        draw_box_end = time.time()

        draw_circles_start = time.time()
        image_circles = copy.deepcopy(image)
        circles_thread = threading.Thread(target=draw_circles, args=[input_json, image_circles,
                                                                     os.path.join(output_path, os.path.basename(output_path) + "_detection_circles.jpeg")])
        circles_thread.start()
        threads.append(circles_thread)
        draw_circles_end = time.time()

    for _thread in threads:
        _thread.join()

    # print("Time taken to draw bboxes: " + str(draw_box_end - draw_box_start))
    # print("Time taken to draw circles: " + str(draw_circles_end - draw_circles_start))

    ExportJsonToCSV(input_json, output_path)
    ExportJsonToText(input_json, output_path)

    return len(input_json), input_json

###############################################################################
# Returns the weight type based on the file extension
#
# Parameters:
# weights_path (str): The path to the weight file
#
# Returns:
# string with the weights type name. if no valid path is supplied, return None
###############################################################################
def GetWeightsType(weights_path):
    if weights_path.endswith(".weights"):
        return "yolo"
    if weights_path.endswith(".pth"):
        return "pytorch"
    if weights_path.endswith(".pb"):
        return "tensorflow"
    if weights_path.endswith("h5"):
        return "keras"

    return None

###############################################################################
# Initialize and returns a tensorflow session. Called inside threads/processes 
# because sessions cannot be pickled. 
#
# Parameters:
# weights_path (str): path to tensorflow weights
#    
# Returns:
# sess (tf.Session): tensorflow session to run detections
###############################################################################
def InitTFSess(weights_path):
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    with tf.gfile.FastGFile(weights_path, 'rb') as f:
       graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7,             
            )
    )
    
    config.intra_op_parallelism_threads = 18
    config.inter_op_parallelism_threads = 18
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    return sess

###############################################################################
# Apply sliding windows on image
#
# Parameters:
# opt_Debug (bool): whether to run the function in debug mode
# image (np array): numpy array representation of image
# progress_Counter (int): progress counter from 0 to 100
# classes (list): image classes to be detected
# opt_img_size (int): the size of the image or image tiles
# opt_window_size (int): dimensions of the sliding windows
# opt_conf_thres (int): confidence threshold for detections
# opt_nms_thres (int): non-maximal suppression threshold for detections
# opt_weights_path (str): path to the trained weights
# output_path (str): path to write detection outputs
# opt_x_stride (int): sliding window stride in x axis
# opt_y_stride (int): sliding window stride in y axis
###############################################################################
def sliding_windows(opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres,
                    opt_nms_thres, opt_weights_path, output_path, opt_x_stride, opt_y_stride):
    
    image = np.asarray(image)
    window_idx = 0
    box_idx = 0
    output_json = {}
    [winW, winH] = [opt_window_size, opt_window_size]

    if opt_Debug:
        if os.path.exists(os.path.join(output_path, "detections.txt")):
            os.remove(os.path.join(output_path, "detections.txt"))

        fp = open(os.path.join(output_path, "detections.txt"), "a")

        if not os.path.isdir(os.path.join(output_path, "sliding_windows")):
            os.mkdir(os.path.join(output_path, "sliding_windows"))

        current_BGW_Path = os.path.join(output_path, "sliding_windows")

    progress_percentage_for_SW = 45
    im_w = (image.shape[1])
    im_h = (image.shape[0])
    total_Win_Count = int(math.modf(im_w / (winW - (winW - (opt_x_stride))))[1]) * int(math.modf(im_h / (winH - (winH - (opt_y_stride))))[1])

    if GetWeightsType(opt_weights_path) == "tensorflow":
        tf_session = InitTFSess(opt_weights_path)

    x_coord = 0
    y_coord = 0

    inference_start = time.time()

    for (x_Offset, y_Offset, window, x_coord, y_coord) in sliding_window(image, x_stepSize=opt_x_stride, y_stepSize=opt_y_stride,
                                                           windowSize=[winW, winH], x_coord=x_coord, y_coord=y_coord):
        progress_Counter = progress_Counter + (progress_percentage_for_SW / total_Win_Count)
        printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)
        window_name = "window_" + str(x_coord) + "_" + str(y_coord)
        window_image = Image.fromarray(window, 'RGB')
        window_width, window_height = window_image.size

        if window_width<winW or window_height < winH or window is None:
            continue

        if not IsBackgroundMostlyBlack(window, window_width, window_height):
            
            if GetWeightsType(opt_weights_path) == "yolo" or GetWeightsType(opt_weights_path) == "pytorch":
                detections = detect_image(opt_window_size, opt_conf_thres, opt_nms_thres, window_image, model)

            if GetWeightsType(opt_weights_path) == "tensorflow":
                detections = detect_image_tensorflow(opt_window_size, opt_conf_thres, opt_nms_thres, window, tf_session)

            if GetWeightsType(opt_weights_path) == None:
                print("Error: No valid weights found. Please supply a valid trained weights for detection.")
                sys.exit()

            if window_width != winW or window_height != winH:
                print("Non-square detection window detected. Window dimension: (" + str(window_width) + ", " + str(window_height) + ")")

            if detections is not None:
                if GetWeightsType(opt_weights_path) == "yolo" or GetWeightsType(opt_weights_path) == 'pytorch':
                    detections = rescale_boxes(detections, opt_img_size, [window_width, window_height])

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    box_name = "box" + str(box_idx)
                    box_w = x2 - x1
                    box_h = y2 - y1
                   
                    if (box_name not in output_json):
                        output_json[box_name] = \
                            {
                                "x1": round(x1.item() + x_Offset),
                                "y1": round(y1.item() + y_Offset),
                                "x2": round(x2.item() + x_Offset),
                                "y2": round(y2.item() + y_Offset),
                                "x1_og": round(x1.item()),
                                "y1_og": round(y1.item()),
                                "x2_og": round(x2.item()),
                                "y2_og": round(y2.item()),
                                "width": round(box_w.item()),
                                "height": round(box_h.item()),
                                "center_x": round(((round (x1.item()) +  x_Offset) + (round (x2.item()) +  x_Offset)) / 2),
                                "center_y": round(((round (y1.item()) +  y_Offset) + (round (y2.item()) +  y_Offset)) / 2),
                                "window_width": window_width,
                                "window_height": window_height,
                                "x_offset": 0,
                                "y_offset": 0,
                                "scaling": 1,
                                "conf": round(conf.item(), 3),
                                "cls_conf": round(cls_conf.data.tolist(), 3),
                                "cls_pred": classes[int(cls_pred)],
                                "model": opt_weights_path,
                                "thread_id": threading.get_ident(),
                                "window_idx": window_name
                            }
                            
                    if opt_Debug:
                        fp.write(classes[int(cls_pred)] + " " + str(round(cls_conf.data.tolist(), 3)) + " "
                                 + str(round(x1.item())) + " " + str(round(y1.item())) + " " + str(round(x2.item())) + " " + str(round(y2.item())) + "\n")

                    box_idx += 1

            window_idx += 1

    inference_end = time.time()
    
    # print("Time taken for inferencing: " + str(inference_end - inference_start) + "")
    t_Table.append(['Total Inferencing Time (Thread ' + str(threading.get_ident()) + ")  ", (inference_end - inference_start)])

    if opt_Debug:
        fp.close()

   
    with open(os.path.join(output_path, "detection.json"), "w") as output_pt:
        json.dump(output_json, output_pt, indent=4)

    obj_no, obj_json = GenerateDetections(opt_Debug, progress_Counter, image, output_path)

###############################################################################
# 
#
#
###############################################################################
def SplitImageByIdx(opt_Debug, image, split):
    split = int(math.pow(2, (split - 1)))
    image_width, image_height = image.size
    N = round((image_width // split))
    M = round((image_height // split))
    if opt_Debug:
        print("SplitImageByIdx Tile Width: " + str(N))
        print("SplitImageByIdx Tile Height: " + str(M))
    image = np.array(image)
    tiles = []
    offsets = []

    y1 = 0
    y2 = 0
    for i in range(split):
        x2 = 0
        x1 = 0
        y1 = y1 + (i * M)
        y2 = y1 + M
        for j in range(split):
            ss = []
            x1 = x1 + (j * N)
            x2 = x1 + N
            ss = image[y1:y2, x1:x2, :]
            tiles.append(ss)
            offsets.append([j * (N), i * (M)])

            if opt_Debug:
                print(ss.shape)
                print("X1: " + str(x1) + " Y1: " + str(y1))
                print("X2: " + str(x2) + " Y2: " + str(y2))

    return tiles, offsets


###############################################################################
# Splits an image into two halves
#
# Parameters:
# image (np array): numpy array representation of image

# Return:
# tiles (list): numpy array represenation of image tiles
# offsets (list): image tile offsets relative to (0,0)
###############################################################################
def SplitImageByHalf(image):
    image_width, image_height = image.size
    image = np.array(image)
    tiles = []
    offsets = []

    M = image_width // 2
    tiles.append(image[:, :M, :])
    tiles.append(image[:, M+1:, :])
    offsets.append([0,0])
    offsets.append([int(image_width // 2), 0])
    return tiles, offsets

###############################################################################
# Splits an image into 8 parts, in a 4 x 2 manner.
#
# Parameters:
# image (np array): numpy array representation of image
#
# Returns:
# tiles (list): numpy array representation of image tiles
# offsets (list): image tile offsets relative to the (0, 0)
###############################################################################
def SplitImageByEight(image):
    image_width, image_height = image.size
    image = np.array(image)
    tiles = []
    offsets = []
    
    M = (image_width // 4) - 1
    N = (image_height // 2) - 1
    
    x1 = y1 = x2 = y2 = 0
    
    for i in range(0, 4):
        x1 = 0
        x2 = 0
        y1 = y1 + (i * M)
        y2 = y1 + M
        for j in range(0, 2):
            ss = []
            x1 = x1 + (j * N)
            x2 = x1 + N
            ss = image[y1:y2, x1:x2, :]
            tiles.append(ss)
            offsets.append([j * (N), i * (M)])
    
    return tiles, offsets

###############################################################################
# checks to see if an exact copy of an array already exist
#
# Parameters:
# myarr (np array): target numpy array
# list_arrays (list): a list of already existing numpy arrays
###############################################################################
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

###############################################################################
# Splits an image according to the split ratio. Adds strides to the edges of the
# image tiles that is half the sliding window width and height.
# 
# Parameters:
# image(numpy array): numpy array representation of the image
# split(int): split ratio 
# winW (int): width of the sliding window
# winH (int): height of the sliding window

# Returns:
# tiles (list): numpy array representations of image tiles
# offsets (list): offsets of each image tile relative to the first image tile
###############################################################################
def SplitImageWithStride(image, split, winW, winH):

    image_width, image_height = image.size

    tile_size_x = image_width // split
    tile_size_y = image_height // split

    tile_stride_x = winW / 2
    tile_stride_y = winH / 2

    tiles = []
    offsets = []

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Image width: " + str(image_width))
    print("Image height: " + str(image_height))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Tile width: " + str(tile_size_x))
    print("Tile height: " + str(tile_size_y))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("X_stride: " + str(tile_stride_x)) 
    print("Y_stride: " + str(tile_stride_y))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    rem_w = image_width % tile_size_x
    rem_h = image_height % tile_size_y
    dev_w = image_width // tile_size_x
    dev_h = image_height // tile_size_y

    y1 = 0
    y2 = 0

    image = np.array(image)

    if image_width <= tile_size_x and image_height <= tile_size_y :
        print(" >>>   1111   >>> ")
        print("X1: " + str(0))
        print("X2: " + str(image_width-1))            
        print("Y1: " + str(0))
        print("Y2: " + str(image_height-1))   
        tiles.append(image)
        offsets.append([0, 0])
    
    elif (image_width <= tile_size_x) and image_height > tile_size_y :
        print(" >>>   222   >>> ")    
        if rem_h >= (tile_size_y/2) or dev_h>1 :
            print(" >>>   222    AA  >>> ")    
            
            split_h = (image_height//tile_size_y)+math.ceil(image_height%tile_size_y/tile_size_y)

            y1 = 0
            y2 = 0
            for i in range(split_h):
                ss = []
                if i == 0 : 
                    y1 = 0 
                else:
                    y1 = y1 + (tile_size_y) - (tile_stride_y)
                   
                y2 = y1 + tile_size_y
                if y2 >= image_height or y2 >=(image_height-(tile_size_y/2)) : 
                    y2 = image_height -1
                
                print("X1: " + str(0))
                print("X2: " + str(image_width-1))            
                print("Y1: " + str(y1))
                print("Y2: " + str(y2))
                ss = image[y1:y2, 0:(image_width-1), :]                    
                tiles.append(ss)
                offsets.append([y1, 0])
    
        else:
            print(" >>>   222    BB  >>> ")    
            print("X1: " + str(0))
            print("X2: " + str(image_width-1))            
            print("Y1: " + str(0))
            print("Y2: " + str(image_height-1))      
            tiles.append(image)
            offsets.append([0, 0])
        
    elif (image_width > tile_size_x) and ( image_height <= tile_size_y ) :
        print("g >>>   333   >>> ")      
        if rem_w >= (tile_size_x / 2) or dev_w>1:
            print(" >>>   333    AA  >>> ")    
            
            split_w = (image_width//tile_size_x)+math.ceil(image_width%tile_size_x/tile_size_x)
            
            x1 = 0
            x2 = 0
            for i in range(split_w):
              
                ss = [] 
#                
                if i == 0 : 
                    x1 = 0 
                else:
                    x1 = x1 + (tile_size_x) - (tile_stride_x)
                                      
                x2 = x1 + tile_size_x
                if x2 >= image_width or x2 >=(image_width-(tile_size_x/2)): 
                    x2 = image_width -1                          
                print("X1: " + str(x1))
                print("X2: " + str(x2))            
                print("Y1: " + str(0))
                print("Y2: " + str(image_height-1))         
                ss = image[0:(image_height-1), x1:x2, :]
                tiles.append(ss)
                offsets.append([0, x1])
                # offsets.append([j * (window_width - x_stride), i * (window_height - y_stride)])
    
        else:
            print(" >>>   333     BB  >>> ")    
            print("X1: " + str(0))
            print("X2: " + str(image_width-1))            
            print("Y1: " + str(0))
            print("Y2: " + str(image_height-1))      

            tiles.append(image)
            offsets.append([0, 0])

    elif (image_width > tile_size_x) and (image_height > tile_size_y):
        print(" >>>   444   >>> ")
        if (rem_w > (tile_size_x / 2) and rem_h > (tile_size_y / 2)) or (dev_w > 1 and dev_h > 1):
            print(" >>>   444  AAA  >>> ")
            y1 = 0
            y2 = 0
            x1 = 0
            x2 = 0
            split_h = (image_height // tile_size_y) + math.ceil(image_height % tile_size_y / tile_size_y)
            split_w = (image_width // tile_size_x) + math.ceil(image_width % tile_size_x / tile_size_x)

            print("Split_H: " + str(split_h))
            print("Split_width: " + str(split_w))

            for i in range(split_h):
                print(" >>>   444  BBB >>> ")
                x2 = 0
                x1 = 0

                if i == 0:
                    y1 = 0
                else:
                    y1 = y1 + (tile_size_y) - (tile_stride_y)

                y2 = y1 + tile_size_y
                if y2 >= image_height or y2 >= (image_height - (tile_size_y / 2)):
                    y2 = image_height - 1

                for j in range(split_w):
                    print(" >>>   444  CCC >>> ")
                    ss = []

                    if j == 0:
                        x1 = 0
                    else:
                        x1 = x1 + (tile_size_x) - (tile_stride_x)

                    x2 = x1 + tile_size_x
                    if x2 >= image_width or x2 >= (image_width - (tile_size_x / 2)):
                        x2 = image_width - 1

                    print("X1: " + str(x1))
                    print("X2: " + str(x2))
                    print("Y1: " + str(y1))
                    print("Y2: " + str(y2))
                    ss = image[int(y1):int(y2), int(x1):int(x2), :]

                    if not arreq_in_list(ss, tiles):
                        tiles.append(ss)
                    
                    if not arreq_in_list([int(j * (tile_size_x - tile_stride_x)), int(i * (tile_size_y - tile_stride_y))], offsets):
                        offsets.append([int(j * (tile_size_x - tile_stride_x)), int(i * (tile_size_y - tile_stride_y))])

    return tiles, offsets
        
###############################################################################
# Save image tiles split from a larger image
# 
# Parameters:
# images (list): a list of numpy array representation of image tiles
###############################################################################
def SaveSplitImages(images):
    image_idx = 0
    for image in images:
        im = Image.fromarray(image, 'RGB')
        im.save(os.path.join(opt_output, os.path.splitext(os.path.basename(opt.image))[0] + "_" + str(image_idx) + ".jpg"))
        image_idx += 1

###############################################################################
# Combines the detection json of each image tile into one detection json.
# 
# Parameters:
# progress_Counter (int): progress bar count from 0 to 100
# output_path (str): path to write the combined detections
# detection_paths (list): a list of paths containing the detections for each image tile
# tile_offsets (list): a list containing the coordinates of tile offsets from a reference tile
        
# Returns:
# returns the string representation of the path of the combined detections
###############################################################################
def CombineDetections(opt_Debug, progress_Counter, output_path, detection_paths, tile_offsets, image_idx):
    detection_jsons = [os.path.join(path, "detection_filtered.json") for path in detection_paths]
    combined_json = {}
    box_idx = 0
    detection_json_index = 0

    image_idx_array = []
    append_count = int(len(detection_jsons) / (image_idx + 1))

    for idx in range(0, image_idx + 1):
        for times in range(0, append_count):
            image_idx_array.append(idx)

    for i in range(0, len(detection_jsons)):
        with open(detection_jsons[i], 'r') as fp:
            detections = json.load(fp)

        for box in detections:
            combined_json["box" + str(box_idx)] = detections[box]
            combined_json["box" + str(box_idx)]["x1"] += tile_offsets[image_idx_array[i]][0]
            combined_json["box" + str(box_idx)]["y1"] += tile_offsets[image_idx_array[i]][1]
            combined_json["box" + str(box_idx)]["x2"] += tile_offsets[image_idx_array[i]][0]
            combined_json["box" + str(box_idx)]["y2"] += tile_offsets[image_idx_array[i]][1]
            box_idx += 1

        detection_json_index += 1

    iou_thres = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    combined_json = SortDetections(opt_Debug, output_path, combined_json)

    for iou in iou_thres:
        filter_bounding_boxes_optimized(opt_Debug, progress_Counter, combined_json, iou)

    with open(os.path.join(output_path, "detection.json"), 'w') as out_fp:
        json.dump(combined_json, out_fp, indent=4)
        
    return os.path.join(output_path, "detection.json")

###############################################################################
# Draws out the combined detections onto the original image
# 
# Parameters:
# output_path (str): progress bar count from 0 to 100
# detection_path (str): path of the combined detections
# image_path (str): path of original image
###############################################################################
def DrawCombineDetections(output_path, detection_path, image_path, color_dict=None):
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    shutil.copyfile(image_path, output_image_path)

    with open(detection_path, 'r') as fp:
        detection = json.load(fp)

    if opt.full_map:
        image = imread(os.path.abspath(image_path), plugin='tifffile')
    else:
        #image = imread(os.path.abspath(image_path), plugin='pil')
        image = Image.open(os.path.abspath(image_path))
    
    draw_bounding_boxes(opt_Debug, detection, image, output_image_path, color_dict)

###############################################################################
# Reads and returns the weights and configuration files.
# 
# Parameters:
# weights_cfg (str): path to file containing weights and config names

# Returns:
# weights (list): a list of weight names
# configs (list): a list of config names
###############################################################################
def ReadConfig(weights_cfg):
    weights = []
    configs = []
    with open(weights_cfg, 'r') as weights_fp:
        for line in weights_fp:
            if line != "":
                if len(line.split(" ")) > 1:
                    weights.append(line.split(" ")[0])
                else:
                    weights.append(line.replace("\n", ""))
            if len(line.split(" ")) > 1:
                if line.split(" ")[1].replace("\n", "").endswith(".cfg"):
                    configs.append(line.split(" ")[1].replace("\n", ""))
    return weights, configs

###############################################################################
# Accepts an image. If image is tiff format, calculate its projections.
# Splits the image into tiles, based on a split ratio.
# Import either PyTorch or Tensorflow according to input weights.
# Starts threads equal to the amount of tiles, and apply threaded sliding windows onto each of the tiles.
# Combine the tile results, and generate a .shp file based on it.
###############################################################################
if __name__ == "__main__":
    main_start = time.time()

    l = 100
    clear()
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    progress_Counter = 1
    printProgressBar(progress_Counter + 1, 100, prefix='Progress:', suffix='Complete', length=50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_cfg", type=str, required=True, help="path to trained weights")
    parser.add_argument("--class_path", type=str, required=True, help="path to class label file")
    parser.add_argument("--image", type=str, required=True, help="the image to apply sliding windows on")
    parser.add_argument("--output", type=str, required=True, help="path to the detections output")
    parser.add_argument("--window_size", type=int, required=True, help="size of the sliding window")
    parser.add_argument("--split", type=int, required=True, help="determines how many sub images to generate")
    
    parser.add_argument("--full_map", type=bool, default=False, help="Run Full Map mode from Tiff")
    parser.add_argument("--Debug", type=bool, default=False, help="Run Full Map mode from Tiff")
    parser.add_argument("--model_def", type=str, help="path to weights cfg file (for certain weights)")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--x_stride", type=int, default=200, help="width stride of the sliding window in pixels")
    parser.add_argument("--y_stride", type=int, default=200, help="height stride of the sliding window in pixels")
    parser.add_argument("--gen_csv", type=bool, default=False, help="whether to generate csv for final result")

    opt = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = 20000000000

    opt_weights_cfg = opt.weights_cfg
    opt_image = opt.image
    opt_output = opt.output
    opt_x_stride = opt.x_stride
    opt_y_stride = opt.y_stride
    opt_full_map = opt.full_map
    opt_Debug = opt.Debug
    opt_window_size = opt.window_size
    opt_conf_thres = opt.conf_thres
    opt_nms_thres = opt.nms_thres
    opt_class_path = opt.class_path
    opt_img_size = opt.img_size
    opt_model_def = opt.model_def

    down_scale = 1
    temp_Folder = "C:\\AiraMapScanner_Temp"

    opt_output = opt.output

    if os.path.exists(opt_output):
        shutil.rmtree(opt_output)
    os.mkdir(opt_output)

    progress_Counter = 5
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    if opt_full_map:
        t_Table.append(['Input Image Type ', 'Full MAP '])
        xOrigin, yOrigin, pixelWidth, pixelHeight, proj_type, jpg_file, im, cord = ConvertFullmap(temp_Folder, opt_image, down_scale)
        opt_image = opt.image
        tiff_path, jpg_extension = os.path.splitext(opt_image)
    else:
        t_Table.append(['Input Image Type ', ' JPG File '])
        t_Table.append(['TiFF Proj Type ', ' None '])
        im = Image.open(opt_image).convert('RGB')

    progress_Counter = 10
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    image_width, image_height = im.size
    image_width = int(round(image_width, -2))
    image_height = int(round(image_height, -2))
    im = im.resize((image_width, image_height))
    im_size = (image_width, image_height)
    t_Table.append(['Input Image Size ', im_size])

    classes = load_classes(opt_class_path)
    image_idx = -1
    threads = []
    output_paths = []
    color_dict = {}

    progress_Counter = 15
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    runtime_start = time.time()
    
    if opt_Debug:
        print("Image width: " + str(image_width) + " Image Height: " + str(image_height))

    sub_images, tile_offsets = SplitImageWithStride(im, opt.split, opt.window_size, opt.window_size)
    weights, configs = ReadConfig(opt.weights_cfg)

    for image in sub_images:
        image_idx += 1
        for weight in weights:
            x_offset = 0
            y_offset = 0
            x_coord = 0
            y_coord = -1

            color_dict[weight] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            output_path = os.path.join(opt_output, os.path.splitext(os.path.basename(opt_image))[0] + "_" + str(image_idx)
                                       + "_" + os.path.splitext(os.path.basename(weight))[0])

            if output_path not in output_paths:
                output_paths.append(output_path)

            if GetWeightsType(weight) == "yolo" or GetWeightsType(weight) == "pytorch":
                from torch.utils.data import DataLoader
                from torchvision import datasets
                from torch.autograd import Variable
            
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                for config in configs:
                    if os.path.splitext(os.path.basename(config))[0] == os.path.splitext(os.path.basename(weight))[0]:
                        config_file = config
            
                model = Darknet(config_file, img_size=opt_img_size).to(device)
            
                if weight.endswith("weights"):
                    print("Loaded the full weights with network architecture.")
                    model.load_darknet_weights(weight)
                else:
                    print("Loaded only the trained weights.")
                    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
            
                if opt_Debug:
                    print("PyTorch model detected.")
                    print("Weights: " + weight + ".")
                    print("Config: " + config_file + ".")
            
                model.eval()
                Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            
                [winW, winH] = [opt_window_size, opt_window_size]
                opt_x_stride = int(winW / 2)
                opt_y_stride = int(winH / 2)
            
                progress_Counter = 30
                printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)
                os.mkdir(output_path)
            
                sliding_windows(opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres, opt_nms_thres,
                                                weight, output_path, opt_x_stride, opt_y_stride)
            
                # child_thread = threading.Thread(target=sliding_windows, args=(opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres, opt_nms_thres,
                #                                 weight, output_path, opt_x_stride, opt_y_stride))
                # child_thread.start()
                # threads.append(child_thread)
            
            elif GetWeightsType(weight) == "tensorflow":
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",category=FutureWarning)
            
                    import tensorflow as tf
            
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
                    if opt_Debug:
                        print("Tensorflow weights detected.")
                        print("Loaded tensorflow weights: " + os.path.basename(weight) + ".")
            
                    [winW, winH] = [opt_window_size, opt_window_size]
                    opt_x_stride = int(winW / 2)
                    opt_y_stride = int(winH / 2)
            
                    progress_Counter = 30
                    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)
            
                    os.mkdir(output_path)
                    
                    # sliding_windows(opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres, opt_nms_thres,
                    # weight, output_path, opt_x_stride, opt_y_stride)
                    
                    child_thread = threading.Thread(target=sliding_windows, args=(
                    opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres, opt_nms_thres,
                    weight, output_path, opt_x_stride, opt_y_stride))
                    
                    child_thread.start()
                    threads.append(child_thread)
            else:
                print("Could not find a valid trained weights for detection. Please supply a valid weights")
                sys.exit()

    for thread in threads:
        thread.join()

    print("Thread count: " + str(len(threads)))

    progress_Counter = 75
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    combined_path = os.path.join(opt.output, "combined_detections")
    os.mkdir(os.path.abspath(combined_path))

    combine_start = time.time()
    combined_json_path = CombineDetections(opt_Debug, progress_Counter, combined_path, output_paths, tile_offsets, image_idx)
    if opt.Debug:
        DrawCombineDetections(combined_path, os.path.join(combined_path, "detection.json"), opt.image, color_dict)
    combine_end = time.time()

    with open(combined_json_path) as fp:
        input_json = json.load(fp)

    gen_Csv = True

    if opt_full_map :
      # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> YESSS TIFFF >>>>>>>>>>>>>>>>")
      ExportJsonToCSV_start = time.time()
      gps_Fields = ExportJsonToCSV2(True, input_json, os.path.join(opt_output, "combined_detections"), xOrigin, yOrigin, pixelWidth, pixelHeight, down_scale)
      # gps_Fields = ExportJsonToCSV2(opt.gen_csv, input_json, opt_output, xOrigin, yOrigin , pixelWidth, pixelHeight, down_scale)
      ExportShpProj(combined_path, gps_Fields, proj_type)
      t_Table.append(['Shp project files  ', tiff_path])
      if gen_Csv:
          ExportJsonToCSV_end = time.time()
          t_Table.append([' Convert Results to CSV  ', (ExportJsonToCSV_end - ExportJsonToCSV_start) ])

    else:
        if gen_Csv :
          ExportJsonToCSV_start = time.time()
          ExportJsonToCSV(input_json, output_path)
          ExportJsonToCSV_end = time.time()
          t_Table.append([' Convert Results to CSV  ', (ExportJsonToCSV_end - ExportJsonToCSV_start) ])

    runtime_end = time.time()

    # Display run stats

    if opt_Debug:
        print("Total runtime: " + str(runtime_end - runtime_start))

    main_end = time.time()
    t_Table.append(['Total Time Elapsed ', (main_end - main_start)])
    progress_Counter = 100
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)
    time.sleep(0.1)

    t = Texttable(180)
    # t.set_cols_width([80,80])
    t.add_rows(t_Table)

    with open(os.path.join(opt.output, "combined_detections", "Summary.txt"), 'a+') as summary:
        summary.write(t.draw())
        
    summary.close()

    print(colored(t.draw(), 'yellow', attrs=['bold']))
    time.sleep(5)
