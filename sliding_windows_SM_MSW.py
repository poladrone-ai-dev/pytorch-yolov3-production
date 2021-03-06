from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from pyimagesearch.find_neighbors import *

from PIL import Image
from skimage.io import imread
from pyimagesearch.helpers import sliding_window
from pyimagesearch.helpers import pyramid
from skimage import io

import pyimagesearch.global_var as global_var
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import sys
import time
import datetime
import argparse
import json
import cv2
import torch
import shutil
import random
import threading
import copy
import glob

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

def detect_image_tensorflow(window, sess=None):
    img_size = 416  # don't change this, because the model is trained on 416 x 416 images
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    rows = opt.window_size
    cols = opt.window_size

    # scale and pad image
    ratio = min(img_size / window.shape[0], img_size / window.shape[1])
    imw = round(window.shape[0] * ratio)
    imh = round(window.shape[1] * ratio)
    inp = window
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

def detect_image(window, model):
    img_size = 416 # don't change this, because the model is trained on 416 x 416 images
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres

    # scale and pad image
    ratio = min(img_size/window.shape[0], img_size/window.shape[1])
    imw = round(window.shape[0] * ratio)
    imh = round(window.shape[1] * ratio)

    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                transforms.Pad((max(int((imh-imw) / 2), 0), max(int((imw-imh) / 2), 0), max(int((imh-imw) / 2), 0), \
                max(int((imw-imh) / 2), 0)), (128, 128, 128)), transforms.ToTensor(), ])

    # convert PIL image to Tensor
    img = Image.fromarray(window)
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    return detections[0]

def SortDetections(output_path, detections):

    detections_list = []

    for box in detections:
        detections_list.append({box: detections[box]})

    sorted_detections = sorted(detections_list, key=lambda item: (list(item.values())[0]['x1'], list(item.values())[0]['y1']))
    sorted_detections_dict = {}

    for i in range(len(sorted_detections)):
        key = list(sorted_detections[i].keys())[0]

        for kvp in sorted_detections[i].values():
            sorted_detections_dict[key] = kvp

    with open(os.path.join(output_path, "detection_sorted.json"), 'w') as json_fp:
        json.dump(sorted_detections_dict, json_fp, indent=4)

    return sorted_detections_dict

def ExportJsonToCSV(detection_json, output_path):

    with open(os.path.join(output_path, "detections.csv"), 'a+') as detection:
        for box in detection_json:
            detection.write(box.replace("box", "") + "," + str(detection_json[box]["conf"]) + ","
                            + str(detection_json[box]["center_x"]) + "," + str(detection_json[box]["center_y"]) + "\n")

def ExportJsonToText(detection_json, output_path):
    with open(os.path.join(output_path, "detections_filtered.txt"), 'a+') as detection:
        for box in detection_json:
            detection.write(box.replace("box", "") + " " + str(detection_json[box]["conf"]) + " " + str(detection_json[box]["x1"]) + " "
                            + str(detection_json[box]["y1"]) + " " + str(detection_json[box]["x2"]) + " "
                            + str(detection_json[box]["y2"]) + '\n')

def filter_bounding_boxes_optimized(detections_json, iou_thres):
    deleted_boxes = []
    same_conf_boxes = []
    num_boxes = len(detections_json)
    print("Number of boxes before filtering: " + str(num_boxes))
    detections_json_list = list(detections_json)

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
                if detections_json[boxA]["conf"] == detections_json[boxB]["conf"] and boxA not in same_conf_boxes and boxB not in same_conf_boxes:
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
    print("Number of deleted boxes: " + str(len(deleted_boxes)))

    for box in list(detections_json):
        if box in deleted_boxes:
            del detections_json[box]

    print("Number of boxes after filtering: " + str(len(detections_json.keys())))
    # print("Reindexing bounding boxes...")
    box_count = 0
    new_detections_json = {}
    for box in detections_json:
        new_detections_json["box" + str(box_count)] = detections_json[box]
        box_count += 1

    return new_detections_json

def calculate_box_offset(output_json, window, box):
    coords = get_tile_coordinates(window)
    output_json[box]["x1"] += coords[1] * opt.x_stride
    output_json[box]["x2"] += coords[1] * opt.x_stride
    output_json[box]["center_x"] += coords[1] * opt.x_stride
    output_json[box]["x_offset"] = coords[1] * opt.x_stride

    output_json[box]["y1"] += coords[0] * opt.y_stride
    output_json[box]["y2"] += coords[0] * opt.y_stride
    output_json[box]["center_y"] += coords[0] * opt.y_stride
    output_json[box]["y_offset"] = coords[0] * opt.y_stride

def draw_bounding_boxes(output_json, image, output_path, shrink_bbox=False, color_dict=None):

    for box in output_json:
        x1 = output_json[box]["x1"]
        y1 = output_json[box]["y1"]
        x2 = output_json[box]["x2"]
        y2 = output_json[box]["y2"]
        width = output_json[box]["width"]
        height = output_json[box]["height"]
        cls_pred = output_json[box]["cls_pred"]
        conf = output_json[box]["conf"]

        if color_dict is not None:
            color = color_dict[output_json[box]["model"]]
        else:
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

def draw_circles(output_json, image, output_path):
    for box in output_json:
        center_x = output_json[box]["center_x"]
        center_y = output_json[box]["center_y"]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 5)

    # cv2.imwrite(output_path, image)
    io.imsave(output_path, image)

def IsBackgroundMostlyBlack(window, winW, winH):
    try:
        if window[int(winW/2), int(winH/2)].all() == np.array([0,0,0]).all():
            return True
    except Exception as e:
        print(e)

    return False

def GenerateDetections(image, output_path):

    threads = []

    if os.path.isfile(os.path.join(output_path, 'detection.json')):
        with open(os.path.join(output_path, 'detection.json'), 'r') as json_file:
            input_json = json.load(json_file)

        image_before_filter = copy.deepcopy(image)
        before_filter_thread = threading.Thread(target=draw_bounding_boxes,
                                                args=[input_json, image_before_filter, os.path.join(output_path,
                                                os.path.basename(output_path) + "_detection_before_filter.jpeg")])
        before_filter_thread.start()
        threads.append(before_filter_thread)
        input_json = SortDetections(output_path, input_json)

        iou_thres_range = [0.5]
        filtering_start = time.time()
        for iou_thres in iou_thres_range:
            input_json = filter_bounding_boxes_optimized(input_json, iou_thres)
        filtering_end = time.time()

        print("Bounding box filtering elapsed time: " + str(filtering_end - filtering_start))

        ExportJsonToCSV(input_json, output_path)
        ExportJsonToText(input_json, output_path)

        with open(os.path.join(output_path, "detection_filtered.json"), "w") as img_json:
            json.dump(input_json, img_json, indent=4)

        image_detect = copy.deepcopy(image)
        draw_box_start = time.time()
        box_thread = threading.Thread(target=draw_bounding_boxes, args=[input_json, image_detect, os.path.join(output_path, os.path.basename(output_path) + "_detection.jpeg")])
        box_thread.start()
        draw_box_end = time.time()

        draw_circles_start = time.time()
        image_circles = copy.deepcopy(image)
        circles_thread = threading.Thread(target=draw_circles, args=[input_json, image_circles, os.path.join(output_path, os.path.basename(output_path) + "_detection_circles.jpeg")])
        circles_thread.start()
        draw_circles_end = time.time()

    for _thread in threads:
        _thread.join()

    # print("Time taken to draw bboxes: " + str(draw_box_end - draw_box_start))
    # print("Time taken to draw circles: " + str(draw_circles_end - draw_circles_start))

    return len(input_json)

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

def sliding_windows(image, window_dim, weights, output_path, x_coord, y_coord, tf_session=None):

    window_idx = 0
    box_idx = 0
    output_json = {}
    [winW, winH] = window_dim

    if os.path.exists(os.path.join(output_path, "detections.txt")):
        os.remove(os.path.join(output_path, "detections.txt"))

    fp = open(os.path.join(output_path, "detections.txt"), "a")

    if not os.path.isdir(os.path.join(output_path, "sliding_windows")):
        os.mkdir(os.path.join(output_path, "sliding_windows"))

    #for resized in pyramid(image, scale=2.0, minSize=windows_minSize):
    for (x, y, window, x_coord, y_coord) in sliding_window(image, x_stepSize=opt.x_stride, y_stepSize=opt.y_stride,
                                                           windowSize=[winW, winH], x_coord=x_coord, y_coord=y_coord):
        window_name = "window_" + str(x_coord) + "_" + str(y_coord)
        window_image = Image.fromarray(window)
        window_width, window_height = window_image.size

        if not IsBackgroundMostlyBlack(window, window_width, window_height):
            window_image.save(os.path.join(output_path, "sliding_windows", window_name + ".jpg"))
            # cv2.imwrite(os.path.join(output_path, "sliding_windows", window_name + ".jpg"), window)
            # io.imsave(os.path.join(output_path, "sliding_windows", window_name + ".jpg"), window)
            print("Performing detection on " + window_name + ".")

            if GetWeightsType(weights) == "yolo" or GetWeightsType(weights) == "pytorch":
                detections = detect_image(window, model)

            if GetWeightsType(weights) == "tensorflow":
                detections = detect_image_tensorflow(window, tf_session)

            if GetWeightsType(weights) == None:
                print("Error: No valid weights found.")
                print("Please supply a valid trained weights for detection.")
                sys.exit()

            if window_width != winW or window_height != winH:
                print("Non-square detection window detected. Window dimension: (" + str(window_width) + ", " + str(window_height) + ")")

            if detections is not None:
                if GetWeightsType(weights) == "yolo" or GetWeightsType(weights) == 'pytorch':
                    detections = rescale_boxes(detections, opt.img_size, window.shape[:2])

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    box_name = "box" + str(box_idx)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    center_x = ((x1.item() + x2.item()) / 2)
                    center_y = ((y1.item() + y2.item()) / 2)

                    if box_name not in output_json:
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
                                "model": weights
                            }

                    fp.write(classes[int(cls_pred)] + " " + str(round(cls_conf.data.tolist(), 3)) + " " + str(round(x1.item()))
                             + " " + str(round(y1.item())) + " " + str(round(x2.item())) + " " + str(round(y2.item())) + "\n")

                    calculate_box_offset(output_json, window_name, box_name)
                    box_idx += 1

            window_idx += 1
            # cv2.waitKey(1)

    fp.close()
    with open(os.path.join(output_path, "detection.json"), "w") as img_json:
        json.dump(output_json, img_json, indent=4)

    GenerateDetections(image, output_path)

def SplitImageByIdx(image, split):
    image_width, image_height = image.size
    N = round((image_width // split) - 1)
    M = round((image_height // split) - 1)
    print(N)
    print(M)
    image = np.array(image)
    tiles = []

    x1 = 0
    x2 = 0
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
            print("X1: " + str(x1) + " Y1: " + str(y1))
            print("X2: " + str(x2) + " Y2: " + str(y2))
            ss = image[y1:y2, x1:x2, :]
            print(ss.shape)
            tiles.append(ss)

    return tiles

def SplitImageByHalf(image):
    image_width, image_height = image.size
    image = np.array(image)
    tiles = []
    M = image_height // 2
    tiles.append(image[:M,:,:])
    tiles.append(image[M+1:,:,:])
    return tiles

def SplitImageWithStride(image, split):
    split = 2
    image_width, image_height = image.size
    window_width = round(image_width // split)
    window_height = round(image_height // split)
    x_stride = 1000 # int(window_width / 10)
    y_stride = 1000 # int(window_height / 10)

    print("Window width: " + str(window_width))
    print("Window height: " + str(window_height))
    print("X_stride: " + str(x_stride))
    print("Y_stride: " + str(y_stride))

    tiles = []
    offsets = []
    image = np.array(image)

    x1 = 0
    x2 = 0
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

def SaveSplitImages(images, image_idx):
    for image in images:
        # io.imsave(os.path.join(opt.output, "img" + str(image_idx) + ".jpg"), image)
        im = Image.fromarray(image)
        im.save(os.path.join(opt.output, "img" + str(image_idx) + ".jpg"))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="path to weights file")
    parser.add_argument("--class_path", type=str, required=True, help="path to class label file")
    parser.add_argument("--image", type=str, required=True, help="the image to apply sliding windows on")
    parser.add_argument("--output", type=str, required=True, help="path to the detections output")
    parser.add_argument("--window_size", type=int, required=True, help="size of the sliding window")
    parser.add_argument("--split", type=int, required=True, help="determines how many sub images to generate")

    parser.add_argument("--model_def", type=str, help="path to weights cfg file (for certain weights)")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--x_stride", type=int, default=200, help="width stride of the sliding window in pixels")
    parser.add_argument("--y_stride", type=int, default=200, help="height stride of the sliding window in pixels")

    Image.MAX_IMAGE_PIXELS = 20000000000

    opt = parser.parse_args()
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.mkdir(opt.output)

    im = Image.open(opt.image).convert('RGB')
    image_width, image_height = im.size

    classes = load_classes(opt.class_path)
    image_idx = 0
    threads = []
    output_paths = []
    runtime_start = time.time()

    print("Image width: " + str(image_width) + " Image Height: " + str(image_height))
    sub_images, tile_offsets = SplitImageWithStride(im, 2)

    # SaveSplitImages(sub_images, image_idx)

    # sub_images = [Image.open(path).convert('RGB') for path in glob.glob(os.path.join(r'test4_with_stride', "*.jpg"))]
    # sub_images = [Image.open(path).convert('RGB') for path in glob.glob(os.path.join(opt.output, "*.jpg"))]
    # sub_images = [np.array(image) for image in sub_images]

    # images_from_disk = [Image.open(path).convert('RGB') for path in glob.glob(os.path.join(opt.output, "*.jpg"))]
    # images_from_disk = [np.array(image) for image in images_from_disk]

    # print(CompareImages(sub_images[0], images_from_disk[0]))

    for image in sub_images:
        x_offset = 0
        y_offset = 0
        x_coord = 0
        y_coord = -1

        if GetWeightsType(opt.weights) == "yolo" or GetWeightsType(opt.weights) == "pytorch":
            from torch.utils.data import DataLoader
            from torchvision import datasets
            from torch.autograd import Variable

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

            print("PyTorch model detected.")

            if opt.weights.endswith("weights"):
                print("Loaded the full weights with network architecture.")
                model.load_darknet_weights(opt.weights)
            else:
                print("Loaded only the trained weights.")
                model.load_state_dict(torch.load(opt.weights, map_location=torch.device('cpu')))

            print("Weights: " + opt.weights + ".")
            print("Config: " + opt.model_def + ".")

            model.eval()  # Set in evaluation mode
            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            [winW, winH] = [opt.window_size, opt.window_size]
            opt.x_stride = int(winW / 2)
            opt.y_stride = int(winH / 2)

            global_var.max_x = (image_width / opt.x_stride) - 1
            global_var.max_y = (image_height / opt.y_stride) - 1

            output_path = os.path.join(opt.output, os.path.splitext(os.path.basename(opt.image))[0] + "_" + str(image_idx))
            image_idx += 1
            output_paths.append(output_path)
            os.mkdir(output_path)

            child_thread = threading.Thread(target=sliding_windows,
                                            args=(image, [winW, winH], opt.weights, output_path, x_coord, y_coord))
            child_thread.start()
            threads.append(child_thread)

        elif GetWeightsType(opt.weights) == "tensorflow":
            import tensorflow as tf

            print("Tensorflow weights detected.")
            print("Loaded tensorflow weights: " + os.path.basename(opt.weights) + ".")

            [winW, winH] = [opt.window_size, opt.window_size]
            opt.x_stride =  int(winW / 2)
            opt.y_stride =  int(winH / 2)
            global_var.max_x = (image_width / opt.x_stride) - 1
            global_var.max_y = (image_height / opt.y_stride) - 1

            with tf.gfile.FastGFile(opt.weights, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            )
            config.gpu_options.allow_growth = True

            sess = tf.Session(config=config)
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            output_path = os.path.join(opt.output, os.path.splitext(os.path.basename(opt.image))[0] + "_" + str(image_idx))
            image_idx += 1
            output_paths.append(output_path)
            os.mkdir(output_path)

            child_thread = threading.Thread(target=sliding_windows, args=(image, [winW, winH], opt.weights, output_path, x_coord, y_coord, sess))
            child_thread.start()
            threads.append(child_thread)
        else:
            print("Could not find a valid trained weights for detection. Please supply a valid weights")
            sys.exit()

    for thread in threads:
        thread.join()

    runtime_end = time.time()
    print("Total runtime: " + str(runtime_end - runtime_start))