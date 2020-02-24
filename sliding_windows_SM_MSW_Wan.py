from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from pyimagesearch.find_neighbors import *

from PIL import Image
from skimage.io import imread
from pyimagesearch.helpers import sliding_window
from pyimagesearch.helpers import pyramid

import pyimagesearch.global_var as global_var
from skimage import io
import os
import sys
import time
import datetime
import argparse
import json
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
import reverse_geocoder as rg
import tifffile as tiff

from termcolor import colored
from texttable import Texttable
import csv
import osgeo.ogr, osgeo.osr
from osgeo import ogr
from osgeo import gdal
from os import system, name

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
#
#
#
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
#
#
#
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
#
#
#
###############################################################################
def detect_image(opt_window_size, opt_conf_thres, opt_nms_thres, window, model):
    img_size = 416  # don't change this, because the model is trained on 416 x 416 images
    conf_thres = opt_conf_thres
    nms_thres = opt_nms_thres

    # scale and pad image
    ratio = min(img_size / window.shape[0], img_size / window.shape[1])
    imw = round(window.shape[0] * ratio)
    imh = round(window.shape[1] * ratio)

    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), \
                                                         max(int((imw - imh) / 2), 0)), (128, 128, 128)),
                                         transforms.ToTensor(), ])

    # convert PIL image to Tensor
    img = Image.fromarray(window, 'RGB')
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    return detections[0]


###############################################################################
#
#
#
###############################################################################
def SortDetections(opt_Debug, opt_output, detections):
    detections_list = []

    for box in detections:
        detections_list.append({box: detections[box]})

    sorted_detections = sorted(detections_list,
                               key=lambda item: (list(item.values())[0]['x1'], list(item.values())[0]['y1']))
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
#
#
#
###############################################################################

def ExportJsonToCSV2(gen_Csv, detection_json, output_path, image_width, image_height, xOrigin, yOrigin, pixelWidth,
                     pixelHeight, rescale):
    csv_file = os.path.splitext(os.path.basename(output_path))[0] + '_detection_filtered.csv'

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
#
#
#
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
#
#
#
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
def ConvertFUllmap(temp_Folder, tiff_Path, down_scale):
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

    cord = reverseGeocode(coordinates)

    t_Table.append(['Map Center Location Lat/Long ', '( ' + str(LAT_center) + ' , ' + str(LONG_center) + ' )'])
    t_Table.append(['Map Location Details ',
                    'City: ' + cord[0]["name"] + ", " + 'State: ' + cord[0]["admin1"] + ", " + 'Country: ' + cord[0]["cc"]])

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
#
#
#
###############################################################################
def ExportShpProj(tiff_path, gps_Fields, proj_type):
    export_shp_proj_start = time.time()

    EPSG_code = proj_type

    shp_file = os.path.splitext(os.path.basename(tiff_path))[0] + '.shp'
    full_shp_path = tiff_path + '\\' + shp_file
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
#
#
#
###############################################################################
def filter_bounding_boxes_optimized(opt_Debug, progress_Counter, image_width, image_height, detections_json, iou_thres):
    bounding_boxes_filter_start = time.time()

    deleted_boxes = []
    same_conf_boxes = []
    num_boxes = len(detections_json)
    if opt_Debug:
        print("Number of boxes before filtering: " + str(num_boxes))
    detections_json_list = list(detections_json)
    # print (len(detections_json_list))

    progress_percentage_for_filtering = 10
    total_Box_Count = len(detections_json_list)

    for idx in range(len(detections_json_list)):

        progress_Counter = progress_Counter + (progress_percentage_for_filtering / total_Box_Count)
        printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

        neighbor_boxes = []
        neighbor_range = 101
        for neighbor_idx in range(1, neighbor_range):
            if idx + neighbor_idx < len(detections_json):
                neighbor_boxes.append(detections_json_list[idx + neighbor_idx])

        for box in neighbor_boxes:
            boxA = detections_json_list[idx]
            boxB = box

            iou, interArea, boxAArea, boxBArea = calculate_iou(detections_json[boxA], detections_json[boxB])

            if iou > iou_thres:
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

    # print("Deleted boxes: " + str(deleted_boxes))
    if opt_Debug:
        print("Number of deleted boxes: " + str(len(deleted_boxes)))

    for box in list(detections_json):
        if box in deleted_boxes:
            del detections_json[box]

    if opt_Debug:
        print("Number of boxes after filtering: " + str(len(detections_json.keys())))

    # print("Reindexing bounding boxes...")

    box_count = 0
    new_detections_json = {}
    for box in detections_json:

        if (int(detections_json[box]["x2"]) < (image_width - 1)) and (
                int(detections_json[box]["y2"]) < (image_height - 1)):
            new_detections_json["box" + str(box_count)] = detections_json[box]
            box_count += 1

    bounding_boxes_filter_end = time.time()

    t_Table.append(['Total Number of Trees Found  ', (len(new_detections_json.keys()) + 1)])

    return new_detections_json

###############################################################################
#
#
#
###############################################################################
def calculate_box_offset(opt_x_stride, opt_y_stride, output_json, window, box):
    coords = get_tile_coordinates(window)
    output_json[box]["x1"] += coords[1] * opt_x_stride
    output_json[box]["x2"] += coords[1] * opt_x_stride
    output_json[box]["center_x"] += coords[1] * opt_x_stride
    output_json[box]["x_offset"] = coords[1] * opt_x_stride

    output_json[box]["y1"] += coords[0] * opt_y_stride
    output_json[box]["y2"] += coords[0] * opt_y_stride
    output_json[box]["center_y"] += coords[0] * opt_y_stride
    output_json[box]["y_offset"] = coords[0] * opt_y_stride


###############################################################################
#
#
#
###############################################################################
def draw_bounding_boxes(output_json, image, output_path, shrink_bbox=False):
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
        color = (255, 0, 0)

        if shrink_bbox:
            x1 += int(0.2 * width)
            y1 += int(0.2 * height)
            x2 -= int(0.2 * width)
            y2 -= int(0.2 * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, box + "-" + str(conf), (int(x1), int(y1)), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    # cv2.imwrite(output_path, image)
    io.imsave(output_path, image)

    draw_bounding_boxes_end = time.time()
    # t_Table.append(['Drawing Results -- Boxes  ', (draw_bounding_boxes_end - draw_bounding_boxes_start) ])

###############################################################################
#
#
#
###############################################################################
def draw_circles(output_json, image, output_path):
    draw_circles_start = time.time()
    for box in output_json:
        center_x = output_json[box]["center_x"]
        center_y = output_json[box]["center_y"]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 5)

    # cv2.imwrite(output_path, image)
    io.imsave(output_path, image)

    draw_circles_end = time.time()
    # t_Table.append(['Drawing Results -- Circles  ', (draw_circles_end - draw_circles_start) ])


###############################################################################
#
#
#
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
#
#
#
###############################################################################
def GenerateDetections(opt_Debug, progress_Counter, image, output_path):
    threads = []

    if os.path.isfile(os.path.join(output_path, 'detection.json')):
        with open(os.path.join(output_path, 'detection.json'), 'r') as json_file:
            input_json = json.load(json_file)

        image_before_filter = copy.deepcopy(image)
        before_filter_thread = threading.Thread(target=draw_bounding_boxes,
                                                args=[input_json, image_before_filter, os.path.join(output_path,
                                                                                                    os.path.basename(
                                                                                                        output_path) + "_detection_before_filter.jpeg")])

        before_filter_thread.start()
        threads.append(before_filter_thread)
        input_json = SortDetections(opt_Debug, output_path, input_json)

        # image_width, image_height = image.size

        image_width = (image.shape[1])
        image_height = (image.shape[0])

        iou_thres_range = [0.5]
        filtering_start = time.time()
        for iou_thres in iou_thres_range:
            input_json = filter_bounding_boxes_optimized(opt_Debug, progress_Counter, image_width, image_height,
                                                         input_json, iou_thres)

        filtering_end = time.time()

        print("Bounding box filtering elapsed time: " + str(filtering_end - filtering_start))

        ExportJsonToCSV(input_json, output_path)
        ExportJsonToText(input_json, output_path)

        with open(os.path.join(output_path, "detection_filtered.json"), "w") as img_json:
            json.dump(input_json, img_json, indent=4)

        image_detect = copy.deepcopy(image)
        draw_box_start = time.time()
        box_thread = threading.Thread(target=draw_bounding_boxes, args=[input_json, image_detect,
                                                                        os.path.join(output_path, os.path.basename(
                                                                            output_path) + "_detection.jpeg")])
        box_thread.start()
        draw_box_end = time.time()

        draw_circles_start = time.time()
        image_circles = copy.deepcopy(image)
        circles_thread = threading.Thread(target=draw_circles, args=[input_json, image_circles,
                                                                     os.path.join(output_path, os.path.basename(
                                                                         output_path) + "_detection_circles.jpeg")])
        circles_thread.start()
        draw_circles_end = time.time()

    for _thread in threads:
        _thread.join()

    # print("Time taken to draw bboxes: " + str(draw_box_end - draw_box_start))
    # print("Time taken to draw circles: " + str(draw_circles_end - draw_circles_start))

    return len(input_json), input_json


###############################################################################
#
#
#
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
#
#
#
###############################################################################


###############################################################################
#
#
#
###############################################################################
# def sliding_windows(image, window_dim, weights, output_path, x_coord, y_coord, tf_session=None):

def sliding_windows(opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres,
                    opt_nms_thres, opt_weights_path, output_path, opt_x_stride, opt_y_stride, window_dim,
                    x_coord, y_coord, tf_session=None):
    # opt_Debug, im,    progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres, opt_nms_thres, opt_weights_path,opt_output,opt_x_stride, opt_y_stride ,opt_image,[winW, winH],x_coord, y_coord, sess
    # opt_nms_thres
    image = np.array(image)
    window_idx = 0
    box_idx = 0
    output_json = {}
    [winW, winH] = window_dim

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
    total_Win_Count = int(math.modf(im_w / (winW - (winW - (opt_x_stride))))[1]) * int(
        math.modf(im_h / (winH - (winH - (opt_y_stride))))[1])

    # for resized in pyramid(image, scale=2.0, minSize=windows_minSize):
    for (x, y, window, x_coord, y_coord) in sliding_window(image, x_stepSize=opt_x_stride, y_stepSize=opt_y_stride,
                                                           windowSize=[winW, winH], x_coord=x_coord, y_coord=y_coord):

        progress_Counter = progress_Counter + (progress_percentage_for_SW / total_Win_Count)
        printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

        if window is None:
            continue

        window_name = "window_" + str(x_coord) + "_" + str(y_coord)
        window_image = Image.fromarray(window, 'RGB')
        window_width, window_height = window_image.size

        if not IsBackgroundMostlyBlack(window, window_width, window_height):
            # window_image.save(os.path.join(output_path, "sliding_windows", window_name + ".jpg"))
            if opt_Debug:
                # print("Performing detection on " + window_name + ".")
                window_image.save(os.path.join(current_BGW_Path, window_name + "_1.jpg"))
                # cv2.imwrite(os.path.join(output_path, "sliding_windows", window_name + ".jpg"), window)
                # io.imsave(os.path.join(output_path, "sliding_windows", window_name + ".jpg"), window)

            hsv = cv2.cvtColor(np.asarray(window_image), cv2.COLOR_BGR2HSV)
            lower_red = np.array([40, 40, 40])
            upper_red = np.array([95, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
            mask1 = cv2.dilate(mask1, np.ones((17, 17), np.uint8), iterations=2)
            res1 = cv2.bitwise_and(np.asarray(window_image), np.asarray(window_image), mask=mask1)
            window = Image.fromarray(res1, 'RGB')
            # window = window_image

            if opt_Debug:
                # print("Performing detection on " + window_name + ".")
                window.save(os.path.join(current_BGW_Path, window_name + "_2.jpg"))
                # cv2.imshow('image',window)
                # cv2.waitKey(3)
                # cv2.imwrite((os.path.join(opt_output, "sliding_windows", window_name + "_2.jpg")),window)

            if GetWeightsType(opt_weights_path) == "yolo" or GetWeightsType(opt_weights_path) == "pytorch":
                detections = detect_image(window, model)

            if GetWeightsType(opt_weights_path) == "tensorflow":
                detections = detect_image_tensorflow(opt_window_size, opt_conf_thres, opt_nms_thres, window, tf_session)

            if GetWeightsType(opt_weights_path) == None:
                print("Error: No valid weights found.")
                print("Please supply a valid trained weights for detection.")
                sys.exit()

            if window_width != winW or window_height != winH:
                print("Non-square detection window detected. Window dimension: (" + str(window_width) + ", " + str(
                    window_height) + ")")

            if detections is not None:
                if GetWeightsType(opt_weights_path) == "yolo" or GetWeightsType(opt_weights_path) == 'pytorch':
                    detections = rescale_boxes(detections, opt_img_size, window.shape[:2])

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    if (round(x2.item()) < (im_w - 5000)) and (round(y2.item()) < (im_h - 5000)):
                        box_name = "box" + str(box_idx)
                        box_w = x2 - x1
                        box_h = y2 - y1
                        center_x = ((x1.item() + x2.item()) / 2)
                        center_y = ((y1.item() + y2.item()) / 2)

                        if (box_name not in output_json):
                            output_json[box_name] = \
                                {
                                    "x1": round(x1.item()),
                                    "y1": round(y1.item()),
                                    "x2": round(x2.item()),
                                    "y2": round(y2.item()),
                                    "x1_og": round(x1.item()),
                                    "y1_og": round(y1.item()),
                                    "x2_og": round(x2.item()),
                                    "y2_og": round(y2.item()),
                                    "width": round(box_w.item()),
                                    "height": round(box_h.item()),
                                    "center_x": round(center_x),
                                    "center_y": round(center_y),
                                    "window_width": window_width,
                                    "window_height": window_height,
                                    "x_offset": x_offset,
                                    "y_offset": y_offset,
                                    "scaling": 1,
                                    "conf": round(conf.item(), 3),
                                    "cls_conf": round(cls_conf.data.tolist(), 3),
                                    "cls_pred": classes[int(cls_pred)],
                                    "model": opt_weights_path
                                }
                        if opt_Debug:
                            fp.write(classes[int(cls_pred)] + " " + str(round(cls_conf.data.tolist(), 3)) + " " + str(
                                round(x1.item()))
                                     + " " + str(round(y1.item())) + " " + str(round(x2.item())) + " " + str(
                                round(y2.item())) + "\n")

                        calculate_box_offset(opt_x_stride, opt_y_stride, output_json, window_name, box_name)
                        box_idx += 1

            window_idx += 1

    if opt_Debug:
        fp.close()

    if opt_Debug:
        with open(os.path.join(output_path, "detection.json"), "w") as img_json:
            json.dump(output_json, img_json, indent=4)

    obj_no, obj_json = GenerateDetections(opt_Debug, progress_Counter, image, output_path)

    # return obj_json


###############################################################################
#
#
#
###############################################################################
def SplitImageByIdx(opt_Debug, image, split):
    split = int(math.pow(2, (split - 1)))
    image_width, image_height = image.size
    N = round((image_width // split) - 1)
    M = round((image_height // split) - 1)
    if opt_Debug:
        print("SplitImageByIdx Tile Width: " + str(N))
        print("SplitImageByIdx Tile Height: " + str(M))
    image = np.array(image)
    tiles = []

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

            if opt_Debug:
                print(ss.shape)
                print("X1: " + str(x1) + " Y1: " + str(y1))
                print("X2: " + str(x2) + " Y2: " + str(y2))

    return tiles


###############################################################################
#
#
#
###############################################################################
def SplitImageByHalf(image):
    image_width, image_height = image.size
    image = np.array(image)
    tiles = []
    M = image_height // 2
    tiles.append(image[:M, :, :])
    tiles.append(image[M + 1:, :, :])
    return tiles


###############################################################################
#
#
#
###############################################################################

def SplitImageWithStride(image, split):
    # split = 0
    # tile_size = 10000
    # tile_stride = 1000
    #
    # image_width, image_height = image.size
    #
    # tiles = []
    # offsets = []
    # image = np.array(image)
    #
    # rem_w = image_width%tile_size
    # rem_h = image_height%tile_size
    #
    # if image_width <= tile_size and image_height <= tile_size :
    #     tiles.append(image)
    #     return  tiles
    #
    # elif (image_width <= tile_size or rem_w <= (tile_size / 2)) and image_height > tile_size :
    #
    #     if rem_h > (tile_size/2) :
    #         split = round (image_height/tile_size)
    #         y1 = 0
    #         y2 = 0
    #         for i in range(split):
    #             ss = []
    #             y1 = y1 + (i * window_height) - (i * y_stride)
    #             y2 = y1 + window_height
    #             ss = image[y1:y2, :, :]
    #             tiles.append(ss)
    #             #offsets.append([j * (window_width - x_stride), i * (window_height - y_stride)])
    #
    #         return tiles
    #     else:
    #         tiles.append(image)
    #         return tiles
    #
    #
    # elif image_width > tile_size and ( image_height <= tile_size or rem_h <= (tile_size / 2)) :
    #
    #     if rem_w > (tile_size / 2):
    #         split = round(image_width / tile_size)
    #         x1 = 0
    #         x2 = 0
    #         for i in range(split):
    #             ss = []
    #             x1 = x1 + (i * window_height) - (i * y_stride)
    #             x2 = x1 + window_height
    #             ss = image[:, x1:x2, :]
    #             tiles.append(ss)
    #             # offsets.append([j * (window_width - x_stride), i * (window_height - y_stride)])
    #
    #         return tiles
    #     else:
    #         tiles.append(image)
    #         return tiles
    #
    # elif (image_width > tile_size and rem_w > (tile_size / 2)) and (image_height > tile_size and rem_h > (tile_size / 2)):
    #
    #     if rem_w > (tile_size / 2) and rem_h > (tile_size / 2):
    #         y1 = 0
    #         y2 = 0
    #
    #         for i in range(split):
    #             x2 = 0
    #             x1 = 0
    #             y1 = y1 + (i * window_height) - (i * y_stride)
    #             y2 = y1 + window_height
    #
    #             for j in range(split):
    #                 ss = []
    #                 x1 = x1 + (j * window_width) - (j * x_stride)
    #                 x2 = x1 + window_width
    #                 ss = image[y1:y2, x1:x2, :]
    #                 tiles.append(ss)
    #                #offsets.append([j * (window_width - x_stride), i * (window_height - y_stride)])
    #
    #         return tiles
    #

#########################################################
    image_width, image_height = image.size
    window_width = round(image_width // split)
    window_height = round(image_height // split)
    x_stride = 1000  # int(window_width / 10)
    y_stride = 1000  # int(window_height / 10)

    print("Window width: " + str(window_width))
    print("Window height: " + str(window_height))
    print("X_stride: " + str(x_stride))
    print("Y_stride: " + str(y_stride))

    tiles = []
    offsets = []
    image = np.array(image)

    y1 = 0
    y2 = 0
    for i in range(split):
        x2 = 0
        x1 = 0
        y1 = y1 + (i * window_height) - (i * y_stride)
        y2 = y1 + window_height

        for j in range(split):
            ss = []
            x1 = x1 + (j * window_width) - (j * x_stride)
            x2 = x1 + window_width
            ss = image[y1:y2, x1:x2]
            tiles.append(ss)
            offsets.append([j * (window_width - x_stride), i * (window_height - y_stride)])

    return tiles, offsets

###############################################################################
#
#
#
###############################################################################
def SaveSplitImages(images, image_idx):
    for image in images:
        # io.imsave(os.path.join(opt_output, "img" + str(image_idx) + ".jpg"), image)
        im = Image.fromarray(image, 'RGB')
        im.save(os.path.join(opt_output, "img" + str(image_idx) + ".jpg"))
        image_idx += 1


def CheckTolerance(pixel1, pixel2):
    print("pixel1: " + str(pixel1))
    print("pixel2: " + str(pixel2))
    if abs(pixel1 - pixel2) <= 20:
        return True
    print("Found two different pixels.")
    return False


def CompareImages(image1, image2):
    if image1.shape != image2.shape:
        return False

    im_height, im_width = image1.shape[0], image1.shape[1]

    thres = 0

    for y in range(0, im_height):
        for x in range(0, im_width):

            if thres > 50:
                return False

            image1_pixel = image1[x, y]
            image2_pixel = image2[x, y]

            if CheckTolerance(image1_pixel[0], image2_pixel[0]) and CheckTolerance(image1_pixel[1], image2_pixel[1]) and \
                    CheckTolerance(image1_pixel[2], image2_pixel[2]):
                pass
            else:
                thres += 1

    return True

def CombineDetections(progress_Counter, image_width, image_height, output_path, detection_paths, tile_offsets):
    detection_jsons = [os.path.join(path, "detection_filtered.json") for path in detection_paths]
    combined_json = {}
    box_idx = 0
    detection_json_index = 0

    for detection_json in detection_jsons:
        with open(detection_json, 'r') as fp:
            detections = json.load(fp)

        for box in detections:
            combined_json["box" + str(box_idx)] = detections[box]
            combined_json["box" + str(box_idx)]["x1"] += tile_offsets[detection_json_index][0]
            combined_json["box" + str(box_idx)]["y1"] += tile_offsets[detection_json_index][1]
            combined_json["box" + str(box_idx)]["x2"] += tile_offsets[detection_json_index][0]
            combined_json["box" + str(box_idx)]["y2"] += tile_offsets[detection_json_index][1]
            box_idx += 1

        detection_json_index += 1

    iou_thres = [0.5]
    combined_json = SortDetections(opt.Debug, output_path, combined_json)

    for iou in iou_thres:
        filter_bounding_boxes_optimized(opt.Debug, progress_Counter, image_width, image_height, combined_json, iou)

    with open(os.path.join(output_path, "detection.json"), 'w') as out_fp:
        json.dump(combined_json, out_fp, indent=4)

def DrawCombineDetections(output_path, detection_path, image_path):
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    shutil.copyfile(image_path, output_image_path)

    with open(detection_path, 'r') as fp:
        detection = json.load(fp)

    if opt.full_map:
        image = imread(os.path.abspath(image_path), plugin='tifffile')
    else:
        image = imread(os.path.abspath(image_path), plugin='pil')

    draw_bounding_boxes(detection, image, output_image_path)

###############################################################################
#
#
#
###############################################################################

if __name__ == "__main__":
    main_start = time.time()

    l = 100
    clear()
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    progress_Counter = 1
    printProgressBar(progress_Counter + 1, 100, prefix='Progress:', suffix='Complete', length=50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="path to weights file")
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

    opt = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = 20000000000

    opt_weights_path = opt.weights
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

    down_scale = 1
    temp_Folder = "C:\\AiraMapScanner_Temp"

    # tiff_path, jpg_extension = os.path.splitext(opt_image)
    opt_output = opt.output

    if os.path.exists(opt_output):
        shutil.rmtree(opt_output)
    os.mkdir(opt_output)

    progress_Counter = 5
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    if opt_full_map:
        t_Table.append(['Input Image Type ', 'Full MAP '])
        xOrigin, yOrigin, pixelWidth, pixelHeight, proj_type, jpg_file, im, cord = ConvertFUllmap(temp_Folder,
                                                                                                  opt_image, down_scale)
        opt_image = jpg_file
        tiff_path, jpg_extension = os.path.splitext(opt_image)

    else:
        t_Table.append(['Input Image Type ', ' JPG File '])
        t_Table.append(['TiFF Proj Type ', ' None '])
        im = Image.open(opt_image).convert('RGB')

    progress_Counter = 10
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    image_width, image_height = im.size
    # print (im.size)
    # image_width  =  (im.shape[1])
    # image_height = (im.shape[0])

    im_size = (image_width, image_height)
    t_Table.append(['Input Image Size ', im_size])

    image_width = int(round(image_width, -2))
    image_height = int(round(image_height, -2))

    classes = load_classes(opt_class_path)
    image_idx = 0
    threads = []
    output_paths = []

    progress_Counter = 15
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    runtime_start = time.time()
    if opt_Debug:
        print("Image width: " + str(image_width) + " Image Height: " + str(image_height))

    sub_images, tile_offsets = SplitImageWithStride(im, 2)

    # sub_images = SplitImageByIdx('true',im, 1)

    # SaveSplitImages(sub_images, image_idx)

    # sub_images = [Image.open(path).convert('RGB') for path in glob.glob(os.path.join(r'test4_with_stride', "*.jpg"))]
    # sub_images = [Image.open(path).convert('RGB') for path in glob.glob(os.path.join(opt_output, "*.jpg"))]
    # sub_images = [np.array(image) for image in sub_images]

    # images_from_disk = [Image.open(path).convert('RGB') for path in glob.glob(os.path.join(opt_output, "*.jpg"))]
    # images_from_disk = [np.array(image) for image in images_from_disk]

    for image in sub_images:
        x_offset = 0
        y_offset = 0
        x_coord = 0
        y_coord = -1

        if GetWeightsType(opt_weights_path) == "yolo" or GetWeightsType(opt_weights_path) == "pytorch":
            from torch.utils.data import DataLoader
            from torchvision import datasets
            from torch.autograd import Variable

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Darknet(opt_model_def, img_size=opt_img_size).to(device)

            if opt.weights.endswith("weights"):
                print("Loaded the full weights with network architecture.")
                model.load_darknet_weights(opt_weights_path)
            else:
                print("Loaded only the trained weights.")
                model.load_state_dict(torch.load(opt_weights_path, map_location=torch.device('cpu')))

            if opt_Debug:
                print("PyTorch model detected.")
                print("Weights: " + opt_weights_path + ".")
                print("Config: " + opt_model_def + ".")

            model.eval()  # Set in evaluation mode
            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            [winW, winH] = [opt_window_size, opt_window_size]
            opt_x_stride = int(winW / 2)
            opt_y_stride = int(winH / 2)

            global_var.max_x = (image_width / opt_x_stride) - 1
            global_var.max_y = (image_height / opt_y_stride) - 1
            
            output_path = os.path.join(opt_output,
                                       os.path.splitext(os.path.basename(opt_image))[0] + "_" + str(image_idx))
            image_idx += 1
            output_paths.append(output_path)
            os.mkdir(output_path)
            
            child_thread = threading.Thread(target=sliding_windows,
                                            args=(image, [winW, winH], opt_weights_path, output_path, x_coord, y_coord))
            child_thread.start()
            threads.append(child_thread)

        elif GetWeightsType(opt_weights_path) == "tensorflow":
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=FutureWarning)
                
                import tensorflow as tf

                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

                if opt_Debug:
                    print("Tensorflow weights detected.")
                    print("Loaded tensorflow weights: " + os.path.basename(opt_weights_path) + ".")

                [winW, winH] = [opt_window_size, opt_window_size]
                opt_x_stride = int(winW / 2)
                opt_y_stride = int(winH / 2)
                global_var.max_x = (image_width / opt_x_stride) - 1
                global_var.max_y = (image_height / opt_y_stride) - 1

                with tf.gfile.FastGFile(opt_weights_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())

                config = tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
                )
                config.gpu_options.allow_growth = True

                sess = tf.Session(config=config)
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

                progress_Counter = 30
                printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

                output_path = os.path.join(opt_output, os.path.splitext(os.path.basename(opt_image))[0] + "_" + str(image_idx))
                image_idx += 1
                output_paths.append(output_path)
                os.mkdir(output_path)

                child_thread = threading.Thread(target=sliding_windows, args=(
                opt_Debug, image, progress_Counter, classes, opt_img_size, opt_window_size, opt_conf_thres, opt_nms_thres,
                opt_weights_path, output_path, opt_x_stride, opt_y_stride, [winW, winH], x_coord, y_coord, sess))
                child_thread.start()
                threads.append(child_thread)

        else:
            print("Could not find a valid trained weights for detection. Please supply a valid weights")
            sys.exit()

    
    for thread in threads:
        thread.join()

    progress_Counter = 75
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)

    combined_path = os.path.join(opt.output, "combined_detections")
    # if os.path.isdir(combined_path):
    #     shutil.rmtree(combined_path)

    os.mkdir(os.path.abspath(combined_path))

    CombineDetections(progress_Counter, image_width, image_height, combined_path, output_paths, tile_offsets)
    combine_start = time.time()
    DrawCombineDetections(combined_path, os.path.join(combined_path, "detection.json"), opt.image)
    combine_end = time.time()

    runtime_end = time.time()

    if opt_Debug:
        print("Total runtime: " + str(runtime_end - runtime_start))

    main_end = time.time()
    t_Table.append(['      Total Time  Elapsed     ', (main_end - main_start)])
    progress_Counter = 100
    printProgressBar(progress_Counter, 100, prefix='Progress:', suffix='Complete', length=50)
    time.sleep(0.1)

    t = Texttable(180)
    # t.set_cols_width([80,80])
    t.add_rows(t_Table)

    txt_summary = os.path.splitext(os.path.basename(tiff_path))[0] + '_Summary.txt'
    with open(os.path.join(output_path, txt_summary), 'a+') as summary:
        summary.write(t.draw())
    summary.close()

    print(colored(t.draw(), 'yellow', attrs=['bold']))
    time.sleep(5)
