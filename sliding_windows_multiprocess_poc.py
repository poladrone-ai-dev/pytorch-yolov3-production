# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:39:14 2020

@author: Z
"""

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
import cv2
import torch
import shutil
import threading
import copy
import pathos.helpers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import pathos.multiprocessing as mp
# import multiprocessing as mp
import gevent
from gevent.threadpool import ThreadPool
import numpy as np
import tensorflow as tf
import warnings
import dill

def draw_bounding_boxes(output_json, image, output_path):
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

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, box + "-" + str(conf), (int(x1), int(y1)), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    io.imsave(output_path, image)
    draw_bounding_boxes_end = time.time()
    # t_Table.append(['Drawing Results -- Boxes  ', (draw_bounding_boxes_end - draw_bounding_boxes_start) ])

def draw_circles(output_json, image, output_path):
    draw_circles_start = time.time()
    for box in output_json:
        center_x = output_json[box]["center_x"]
        center_y = output_json[box]["center_y"]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 5)

    io.imsave(output_path, image)
    draw_circles_end = time.time()
    # t_Table.append(['Drawing Results -- Circles  ', (draw_circles_end - draw_circles_start) ])

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

def filter_bounding_boxes_optimized(opt_Debug, image_width, image_height, detections_json, iou_thres):
    bounding_boxes_filter_start = time.time()

    deleted_boxes = []
    same_conf_boxes = []
    num_boxes = len(detections_json)
    
    if opt_Debug:
        print("Number of boxes before filtering: " + str(num_boxes))
        
    detections_json_list = list(detections_json)

    total_Box_Count = len(detections_json_list)
    

    for idx in range(len(detections_json_list)):

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

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def SaveSplitImages(images, output, image_path):
    image_idx = 0
    for image in images:
        im = Image.fromarray(image, 'RGB')
        im.save(os.path.join(output, os.path.splitext(os.path.basename(image_path))[0] + "_" + str(image_idx) + ".jpg"))
        image_idx += 1
        
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

def GenerateDetections(opt_Debug, image, output_path):
    threads = []

    if os.path.isfile(os.path.join(output_path, 'detection.json')):
        with open(os.path.join(output_path, 'detection.json'), 'r') as json_file:
            input_json = json.load(json_file)

        image = np.asarray(image)

        image_before_filter = copy.deepcopy(image)
        before_filter_thread = threading.Thread(target=draw_bounding_boxes,
                                                args=[input_json, image_before_filter,
                                                        os.path.join(output_path,os.path.basename(output_path) + "_detection_before_filter.jpeg")])

        before_filter_thread.start()
        threads.append(before_filter_thread)

        input_json = SortDetections(True, output_path, input_json)

        image_width = (image.shape[1])
        image_height = (image.shape[0])

        iou_thres_range = [0.5]
        filtering_start = time.time()

        for iou_thres in iou_thres_range:
            input_json = filter_bounding_boxes_optimized(opt_Debug, image_width, image_height, input_json, iou_thres)

        filtering_end = time.time()

        print("Bounding box filtering elapsed time: " + str(filtering_end - filtering_start))

        with open(os.path.join(output_path, "detection_filtered.json"), "w") as img_json:
            json.dump(input_json, img_json, indent=4)

        image_detect = copy.deepcopy(image)
        draw_box_start = time.time()
        box_thread = threading.Thread(target=draw_bounding_boxes, args=[input_json, image_detect,
                                                                        os.path.join(output_path, os.path.basename(output_path) + "_detection.jpeg")])
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

    return len(input_json), input_json

def sliding_windows(opt_Debug, image, classes, opt_window_size, opt_conf_thres,
                    opt_nms_thres, opt_weights_path, output_path, opt_x_stride, opt_y_stride, x_coord, y_coord):
    
    import numpy as np
    import os
    import math
    import tensorflow as tf
    from pyimagesearch.helpers import sliding_window
    from PIL import Image
    
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

    im_w = (image.shape[1])
    im_h = (image.shape[0])
    total_Win_Count = int(math.modf(im_w / (winW - (winW - (opt_x_stride))))[1]) * int(math.modf(im_h / (winH - (winH - (opt_y_stride))))[1])

    with tf.gfile.FastGFile(opt_weights_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    )
    
    config.gpu_options.allow_growth = True

    tf_session = tf.Session(config=config)
    tf_session.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for (x_Offset, y_Offset, window, x_coord, y_coord) in sliding_window(image, x_stepSize=opt_x_stride, y_stepSize=opt_y_stride,
                                                           windowSize=[winW, winH], x_coord=x_coord, y_coord=y_coord):
        window_name = "window_" + str(x_coord) + "_" + str(y_coord)
        window_image = Image.fromarray(window, 'RGB')
        window_width, window_height = window_image.size

        if window_width<winW or window_height < winH or window is None:
            continue

        if not IsBackgroundMostlyBlack(window, window_width, window_height):

            detections = detect_image_tensorflow(opt_window_size, opt_conf_thres, opt_nms_thres, window, tf_session)

            if window_width != winW or window_height != winH:
                print("Non-square detection window detected. Window dimension: (" + str(window_width) + ", " + str(window_height) + ")")

            if detections is not None:

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

    if opt_Debug:
        fp.close()

    if opt_Debug:
        with open(os.path.join(output_path, "detection.json"), "w") as img_json:
            json.dump(output_json, img_json, indent=4)

    obj_no, obj_json = GenerateDetections(opt_Debug, image, output_path)


def CombineDetections(image_width, image_height, output_path, detection_paths, tile_offsets):
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
    combined_json = SortDetections(True, output_path, combined_json)

    for iou in iou_thres:
        filter_bounding_boxes_optimized(True, image_width, image_height, combined_json, iou)

    with open(os.path.join(output_path, "detection.json"), 'w') as out_fp:
        json.dump(combined_json, out_fp, indent=4)

def DrawCombineDetections(output_path, detection_path, image_path):
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    shutil.copyfile(image_path, output_image_path)

    with open(detection_path, 'r') as fp:
        detection = json.load(fp)

    image = imread(os.path.abspath(image_path), plugin='pil')
    draw_bounding_boxes(detection, image, output_image_path)

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    Image.MAX_IMAGE_PIXELS = 20000000000
    split = 2
    weights_path = r'checkpoints\frozen_inference_graph_RCNN_BYT_V2_Rev1.pb'
    image_path = r'test_images\test_4.jpg'
    output = r'sliding_windows_output'
    class_path = r'coco_wan.names'
    window_size = 1000
    pathos.helpers.freeze_support()
    
    runtime_start = time.time()
    
    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output)
    
    image = Image.open(image_path).convert('RGB')
    
    image_width, image_height = image.size
    image_width = int(round(image_width, -2))
    image_height = int(round(image_height, -2))
    image = image.resize((image_width, image_height))
    im_size = (image_width, image_height)
    
    classes = load_classes(class_path)
    image_idx = 0
    threads = []
    output_paths = []
    
    sub_images, tile_offsets = SplitImageWithStride(image, split, window_size, window_size)
    # SaveSplitImages(sub_images, output, image_path)
    # sub_images = [Image.open(path) for path in glob.glob(os.path.join(output, "*.jpg"))]
    
    x_stride = window_size / 2
    y_stride = window_size / 2
    
    # pool = mp.Pool(processes=split)
    pool = ThreadPool(split)
    
    with warnings.catch_warnings():
        for image in sub_images:
            x_offset = 0
            y_offset = 0
            x_coord = 0
            y_coord = -1
    
            warnings.filterwarnings("ignore", category=FutureWarning)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            [winW, winH] = [window_size, window_size]
            
            output_path = os.path.join(output, os.path.splitext(os.path.basename(image_path))[0] + "_" + str(image_idx))
            image_idx += 1
            output_paths.append(output_path)
            os.mkdir(output_path)
            
            # result = pool.starmap(sliding_windows, [(True, image, classes, window_size, 0.3, 0.4,
            #                                           weights_path, output_path, int(x_stride), int(y_stride), x_coord, y_coord)])
            
            result = pool.spawn(sliding_windows, True, image, classes, window_size, 0.3, 0.4,
                                                      weights_path, output_path, int(x_stride), int(y_stride), x_coord, y_coord)
    
    gevent.wait()
    
    combined_path = os.path.join(output, "combined_detections")
    os.mkdir(os.path.abspath(combined_path))

    CombineDetections(image_width, image_height, combined_path, output_paths, tile_offsets)
    combine_start = time.time()
    DrawCombineDetections(combined_path, os.path.join(combined_path, "detection.json"), image_path)
    combine_end = time.time()

    runtime_end = time.time()
    print("Total runtime: " + str(runtime_end - runtime_start))
