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
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    with open(os.path.join(output_path, "detections_sorted.json"), 'w') as json_fp:
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
    print("Reindexing bounding boxes...")
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

def draw_bounding_boxes(output_json, image, shrink_bbox=False):
    for box in output_json:
        x1 = output_json[box]["x1"]
        y1 = output_json[box]["y1"]
        x2 = output_json[box]["x2"]
        y2 = output_json[box]["y2"]
        width = output_json[box]["width"]
        height = output_json[box]["height"]
        cls_pred = output_json[box]["cls_pred"]
        conf = output_json[box]["conf"]

        if shrink_bbox:
            x1 += int(0.2 * width)
            y1 += int(0.2 * height)
            x2 -= int(0.2 * width)
            y2 -= int(0.2 * height)

        # if not shrink_bbox:
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # else:
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, box + "-" + str(conf), (int(x1), int(y1)), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, lineType=cv2.LINE_AA)

def draw_circles(output_json, image):
    for box in output_json:
        center_x = output_json[box]["center_x"]
        center_y = output_json[box]["center_y"]
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 5)

def IsBackgroundMostlyBlack(window, winW, winH):
    try:
        if window[int(winW/2), int(winH/2)].all() == np.array([0,0,0]).all():
            return True
    except Exception as e:
        print(e)

    return False

def sliding_windows(window_dim, weights, output_path, tf_session=None):
    image = imread(opt.image, plugin='pil') # specifies pil plugin, or the default one program searches is used (TIFF, etc...)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
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
    for (x, y, window) in sliding_window(image, x_stepSize=opt.x_stride, y_stepSize=opt.y_stride, windowSize=[winW, winH]):

        if window is None:
            continue

        window_name = "window_" + str(global_var.x_coord) + "_" + str(global_var.y_coord)
        window_image = Image.fromarray(window)
        window_width, window_height = window_image.size

        if not IsBackgroundMostlyBlack(window, window_width, window_height):

            window_image.save(os.path.join(output_path, "sliding_windows", window_name + ".jpg"))

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
                                "x_offset": global_var.x_offset,
                                "y_offset": global_var.y_offset,
                                "scaling": 1,
                                "conf": round(conf.item(), 3),
                                "cls_conf": round(cls_conf.data.tolist(), 3),
                                "cls_pred": classes[int(cls_pred)]
                            }

                    fp.write(classes[int(cls_pred)] + " " + str(round(cls_conf.data.tolist(), 3)) + " " + str(round(x1.item()))
                             + " " + str(round(y1.item())) + " " + str(round(x2.item())) + " " + str(round(y2.item())) + "\n")

                    calculate_box_offset(output_json, window_name, box_name)
                    box_idx += 1

            window_idx += 1
            cv2.waitKey(1)

    fp.close()
    cv2.imwrite(os.path.join(output_path, "output.jpeg"), image)
    with open(os.path.join(output_path, "detection.json"), "w") as img_json:
        json.dump(output_json, img_json, indent=4)

def dataloader(window_dim):
    output_json = {}
    [winW, winH] = window_dim
    window_idx = 0
    output_path = opt.output
    box_idx = 0

    image_path = os.path.dirname(opt.image)
    dataloader = DataLoader(
        ImageFolder(image_path, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    if os.path.exists(os.path.join(output_path, "detections.txt")):
        os.remove(os.path.join(output_path, "detections.txt"))

    fp = open(os.path.join(output_path, "detections.txt"), "a")

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        if os.path.basename(opt.image) == os.path.basename(img_paths[0]):
            input_imgs = Variable(input_imgs.type(Tensor))

            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 0.8, 0.4)

            imgs.extend(img_paths)
            img_detections.extend(detections)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        im = Image.open(path)
        image_width, image_height = im.size
        img = np.array(Image.open(path))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # box_idx = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if classes[int(cls_pred)] != "palm0":
                    # if True:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    if x1 < 0:
                        x1 = torch.tensor(0)

                    if y1 < 0:
                        y1 = torch.tensor(0)

                    if x2 > im.width:
                        x2 = torch.tensor(im.width)

                    if y2 > im.height:
                        y2 = torch.tensor(im.height)

                    box_w = x2 - x1
                    box_h = y2 - y1

                    center_x = ((x1.item() + x2.item()) / 2)
                    center_y = ((y1.item() + y2.item()) / 2)

                    box_idx += 1
                    output_json["window0"] = {}
                    output_json["window0"]["box" + str(box_idx)] = {
                        "x1": round(x1.item()),
                        "y1": round(y1.item()),
                        "x2": round(x2.item()),
                        "y2": round(y2.item()),
                        "width": round(box_w.item()),
                        "height": round(box_h.item()),
                        "conf": round(conf.item(), 3),
                        "center_x": round(center_x),
                        "center_y": round(center_y),
                        "cls_conf": round(cls_conf.data.tolist(), 3),
                        "cls_pred": classes[int(cls_pred)]
                    }

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    cv2.imwrite(os.path.join(opt.output,
                                             os.path.basename(path)[:-4] + ".jpg"), img)

                    cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, \
                                1.0, (0, 0, 0), lineType=cv2.LINE_AA)
                    cv2.putText(img, os.path.basename(path), (0, int(im.height / 2)), \
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)

                    fp.write(str(classes[int(cls_pred)]) + " " + str(conf.item()).replace(str(conf)[0], '')
                             + " " + str(int(x1.item())) + " " + str(int(y1.item())) + " "
                             + str(int(box_w.item())) + " " + str(int(box_h.item())) + '\n')

    fp.close()

    with open(os.path.join(output_path, "detection.json"), "a") as img_json:
        json.dump(output_json, img_json, indent=4)

def CombineDetections(output_path, detection_paths, image_path):
    detection_jsons = [os.path.join(path, "detection_filtered.json") for path in detection_paths]
    output_image = os.path.join(output_path, os.path.basename(image_path))

    shutil.copyfile(image_path, output_image)
    shrink_bbox = False

    for detection_json in detection_jsons:
        if os.path.isfile(detection_json):
            with open(detection_json, 'r') as fp:
                detection = json.load(fp)

            image = imread(output_image, plugin='pil')
            draw_bounding_boxes(detection, image, shrink_bbox)
            shrink_bbox = ~shrink_bbox
            cv2.imwrite(output_image, image)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights_path", type=str, required=True, help="path to trained weights")
    parser.add_argument("--class_path", type=str, required=True, help="path to class label file")
    parser.add_argument("--image", type=str, required=True, help="the image to apply sliding windows on")
    parser.add_argument("--output", type=str, required=True, help="path to the detections output")
    parser.add_argument("--window_size", type=int, required=True, help="size of the sliding window")
    parser.add_argument("--second_weight", type=str, required=True, help="second trained weight for comparison")

    parser.add_argument("--model_def", type=str, help="path to model definition file")
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

    im = Image.open(opt.image)
    image_width, image_height = im.size
    image_width = int(round(image_width, -2))
    image_height = int(round(image_height, -2))

    classes = load_classes(opt.class_path)
    weights = [opt.weights_path, opt.second_weight]
    output_paths = []
    shrink_bbox = False
    second_weight = False

    # sliding window threads
    first_thread = None
    second_thread = None

    for weight in weights:
        print("Performing detection on " + opt.image + " with " + os.path.basename(weight) + ".")
        if GetWeightsType(weight) == "yolo" or GetWeightsType(weight) == "pytorch":

            from torch.utils.data import DataLoader
            from torchvision import datasets
            from torch.autograd import Variable

            output_path = os.path.join(opt.output, os.path.splitext(os.path.basename(weight))[0])
            output_paths.append(output_path)
            os.mkdir(output_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
            print("PyTorch model detected.")

            if weight.endswith("weights"):
                print("Loaded the full weights with network architecture.")
                model.load_darknet_weights(weight)
            else:
                print("Loaded only the trained weights.")
                model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))

            # if not second_weight:
            #     if opt.weights_path.endswith("weights"):
            #         print("Loaded the full weights with network architecture.")
            #         model.load_darknet_weights(opt.weights_path)
            #     else:
            #         print("Loaded only the trained weights.")
            #         model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))
            # else:
            #     if opt.second_weight.endswith("weights"):
            #         print("Loaded the full weights with network architecture.")
            #         model.load_darknet_weights(opt.second_weight)
            #     else:
            #         print("Loaded only the trained weights.")
            #         model.load_state_dict(torch.load(opt.second_weight, map_location=torch.device('cpu')))

            print("Weights: " + os.path.basename(weight) + ".")
            print("Config: " + os.path.basename(opt.model_def) + ".")

            model.eval() # Set in evaluation mode

            classes = load_classes(opt.class_path)  # Extracts class labels from file
            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            if image_width <= 500 and image_height <= 500:
                [winW, winH] = [230, 230]
                opt.x_stride = int(winW / 2)
                opt.y_stride = int(winH / 2)
                dataloader([winW, winH])
                global_var.x_offset = -winW
                global_var.y_offset = -winH

            else:
                [winW, winH] = [opt.window_size, opt.window_size]
                opt.x_stride = int(winW / 2)
                opt.y_stride = int(winH / 2)
                global_var.max_x = (image_width / opt.x_stride) - 1
                global_var.max_y = (image_height / opt.y_stride) - 1
                print("Running sliding windows on " + opt.image + ". Window dimension: [" + str(winW) + ", " + str(winH) + "]")
                start_time = time.time()

                first_thread = threading.Thread(target=sliding_windows, args=([winW, winH], weight, output_path)).start()

                # if not second_weight:
                #     # sliding_windows([winW, winH], opt.weights_path, output_path)
                #     first_thread = threading.Thread(target=sliding_windows, args=([winW, winH], opt.weights_path, output_path)).start()
                #     # first_thread = multiprocessing.Process(target=sliding_windows, args=([winW, winH], opt.weights_path, output_path, opt)).start()
                # else:
                #     # sliding_windows([winW, winH], opt.second_weight, output_path)
                #     second_thread = threading.Thread(target=sliding_windows, args=([winW, winH], opt.second_weight, output_path)).start()
                #     # second_thread = multiprocessing.Process(target=sliding_windows, args=([winW, winH], opt.second_weight, output_path, opt)).start()

                end_time = time.time()
                print("Time elapsed for YOLO/PyTorch detection: " + str(end_time - start_time) + "s.")

        elif GetWeightsType(weight) == "tensorflow":
            import tensorflow as tf
            print("Tensorflow weights detected.")
            print("Loaded tensorflow weights: " + os.path.basename(weight) + ".")

            output_path = os.path.join(opt.output, os.path.splitext(os.path.basename(weight))[0])
            output_paths.append(output_path)
            os.mkdir(output_path)
            [winW, winH] = [opt.window_size, opt.window_size]
            opt.x_stride = int(winW / 2)
            opt.y_stride = int(winH / 2)
            global_var.max_x = (image_width / opt.x_stride) - 1
            global_var.max_y = (image_height / opt.y_stride) - 1

            print("Running sliding windows on " + opt.image + ". Window dimension: [" + str(winW) + ", " + str(winH) + "]")
            start_time = time.time()

            with tf.gfile.FastGFile(weight, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # if not second_weight:
            #     with tf.gfile.FastGFile(opt.weights_path, 'rb') as f:
            #         graph_def = tf.GraphDef()
            #         graph_def.ParseFromString(f.read())
            # else:
            #     with tf.gfile.FastGFile(opt.second_weight, 'rb') as f:
            #         graph_def = tf.GraphDef()
            #         graph_def.ParseFromString(f.read())

            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            )
            config.gpu_options.allow_growth = True

            sess = tf.Session(config=config)
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            first_thread = threading.Thread(target=sliding_windows, args=([winW, winH], weight, output_path, sess)).start()

            # if not second_weight:
            #     # sliding_windows([winW, winH], opt.weights_path, output_path, sess)
            #     first_thread = threading.Thread(target=sliding_windows, args=([winW, winH], opt.weights_path, output_path, sess)).start()
            #     # first_thread = multiprocessing.Process(target=sliding_windows, args=([winW, winH], opt.weights_path, output_path, opt, sess)).start()
            # else:
            #     # sliding_windows([winW, winH], opt.second_weight, output_path, sess)
            #     second_thread = threading.Thread(target=sliding_windows, args=([winW, winH], opt.second_weight, output_path, sess)).start()
            #     # second_thread = multiprocessing.Process(target=sliding_windows, args=([winW, winH], opt.second_weight, output_path, opt, sess)).start()

            # sess.close()
            end_time = time.time()
            print("Time elapsed for Tensorflow detection: " + str(end_time - start_time) + "s.")
        else:
            print("Could not find a valid trained weights for detection. Please supply a valid weights")
            sys.exit()

        if os.path.isfile(os.path.join(output_path, 'detection.json')):
            with open(os.path.join(output_path, 'detection.json'), 'r') as json_file:
                input_json = json.load(json_file)

            image_before_filter = imread(opt.image, plugin='pil')
            draw_bounding_boxes(input_json, image_before_filter)
            cv2.imwrite(os.path.join(output_path, os.path.basename(output_path) + "_detection_before_filter.jpeg"), image_before_filter)
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

            image = imread(opt.image, plugin='pil')
            draw_bounding_boxes(input_json, image, shrink_bbox=shrink_bbox)
            cv2.imwrite(os.path.join(output_path, os.path.basename(output_path) + "_detection.jpeg"), image)

            image_circles = imread(opt.image, plugin='pil')
            draw_circles(input_json, image_circles)
            cv2.imwrite(os.path.join(output_path, os.path.basename(output_path) + "_detection_circles.jpeg"), image_circles)

            global_var.x_offset = 0
            global_var.y_offset = 0
            global_var.x_coord = 0
            global_var.y_coord = -1
            shrink_bbox = True
            second_weight = True
            # first_thread.join()

        #end of loop

    # print("Process first_thread is alive: {}".format(first_thread.pid))
    # print("Process second_thread is alive: {}".format(second_thread.pid))

    combined_path = os.path.join(opt.output, "combined_detections")

    if os.path.isdir(combined_path):
        shutil.rmtree(combined_path)

    os.mkdir(combined_path)
    CombineDetections(combined_path, output_paths, opt.image)

    for output_path in output_paths:
        if os.path.isfile(os.path.join(output_path, os.path.basename(output_path) + "_detection_before_filter.jpeg")):
            shutil.copyfile(os.path.join(output_path, os.path.basename(output_path) + "_detection_before_filter.jpeg"),
                            os.path.join(combined_path, os.path.basename(output_path) + "_detection_before_filter.jpeg"))

        if os.path.isfile(os.path.join(output_path, os.path.basename(output_path) + "_detection.jpeg")):
            shutil.copyfile(os.path.join(output_path, os.path.basename(output_path) + "_detection.jpeg"),
                            os.path.join(combined_path, os.path.basename(output_path) + "_detection.jpeg"))
