from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from pyimagesearch.find_neighbors import *

from PIL import Image
from skimage.io import imread
from matplotlib.ticker import NullLocator
from pyimagesearch.helpers import sliding_window
from pyimagesearch.helpers import pyramid
from collections import OrderedDict

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
import pstats

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA["x2"] - boxA["x1"] + 1) * (boxA["y2"] - boxA["y1"] + 1)
    boxBArea = (boxB["x2"] - boxB["x1"] + 1) * (boxB["y2"] - boxB["y1"] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
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

    sw_img = window
    # inp = cv2.resize(sw_img, (imw, imh))

    inp = sw_img

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
            print("Class ID = >>%d" % classId + "  score  =>>> %f" % score)
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

def SortDetections(detections):

    detections_list = []

    for box in detections:
        detections_list.append({box: detections[box]})

    sorted_detections = sorted(detections_list, key=lambda item: (list(item.values())[0]['x1'], list(item.values())[0]['y1']))
    sorted_detections_dict = {}

    for i in range(len(sorted_detections)):
        key = list(sorted_detections[i].keys())[0]

        for kvp in sorted_detections[i].values():
            sorted_detections_dict[key] = kvp

    with open(os.path.join(opt.output, "detections_sorted.json"), 'w') as json_fp:
        json.dump(sorted_detections_dict, json_fp, indent=4)

    return sorted_detections_dict

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

            # print(boxA)
            # print(boxB)
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

    print("Deleted boxes: " + str(deleted_boxes))
    print("Number of deleted boxes: " + str(len(deleted_boxes)))

    for box in list(detections_json):
        if box in deleted_boxes:
            del detections_json[box]

    print("Number of boxes after filtering: " + str(len(detections_json.keys())))
    return detections_json

def filter_bounding_boxes(output_json, iou_thres):
    num_boxes = len(output_json)
    print("Number of boxes before filtering: " + str(num_boxes))
    deleted_boxes = []
    same_conf_boxes = []

    for boxA in output_json:
        for boxB in output_json:
            if boxA != boxB:
                iou, interArea, boxAArea, boxBArea = calculate_iou(output_json[boxA], output_json[boxB])

                if iou > iou_thres:
                    if output_json[boxA]["conf"] == output_json[boxB]["conf"] and boxA not in same_conf_boxes and boxB not in same_conf_boxes:
                        rand_num = random.randint(1,2)

                        if rand_num == 1:
                            if boxA not in deleted_boxes and boxA not in same_conf_boxes:
                                deleted_boxes.append(boxA)
                                same_conf_boxes.append(boxA)
                        elif rand_num == 2:
                            if boxB not in deleted_boxes and boxB not in same_conf_boxes:
                                deleted_boxes.append(boxB)
                                same_conf_boxes.append(boxB)

                    if output_json[boxA]["conf"] < output_json[boxB]["conf"]:
                        if boxA not in deleted_boxes:
                            deleted_boxes.append(boxA)
                    elif output_json[boxB]["conf"] < output_json[boxA]["conf"]:
                        if boxB not in deleted_boxes:
                            deleted_boxes.append(boxB)


    print("Deleted boxes: " + str(deleted_boxes))
    print("Number of deleted boxes: " + str(len(deleted_boxes)))

    for box in list(output_json):
        if box in deleted_boxes:
            del output_json[box]

    print("Number of boxes after filtering: " + str(len(output_json.keys())))
    return output_json

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

def draw_bounding_boxes(output_json, image):
    for box in output_json:
        x1 = output_json[box]["x1"]
        y1 = output_json[box]["y1"]
        x2 = output_json[box]["x2"]
        y2 = output_json[box]["y2"]
        cls_pred = output_json[box]["cls_pred"]
        conf = output_json[box]["conf"]

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

def sliding_windows(window_dim, tf_session=None):
    image = imread(opt.image, plugin='pil') # specifies pil plugin, or the default one program searches is used (TIFF, etc...)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    output_path = opt.output
    window_idx = 0
    box_idx = 0
    output_json = {}
    [winW, winH] = window_dim

    if os.path.exists(os.path.join(output_path, "detections.txt")):
        os.remove(os.path.join(output_path, "detections.txt"))

    fp = open(os.path.join(output_path, "detections.txt"), "a")

    if not os.path.isdir(os.path.join(opt.output, "sliding_windows")):
        os.mkdir(os.path.join(opt.output, "sliding_windows"))

    #for resized in pyramid(image, scale=2.0, minSize=windows_minSize):
    for (x, y, window) in sliding_window(image, x_stepSize=opt.x_stride, y_stepSize=opt.y_stride, windowSize=[winW, winH]):

        if window is None:
            continue

        window_name = "window_" + str(global_var.x_coord) + "_" + str(global_var.y_coord)
        window_image = Image.fromarray(window)
        window_width, window_height = window_image.size

        if not IsBackgroundMostlyBlack(window, window_width, window_height):

            window_image.save(os.path.join(opt.output, "sliding_windows", window_name + ".jpg"))

            print("Performing detection on " + window_name + ".")

            if GetWeightsType() == "yolo" or GetWeightsType() == "pytorch":
                detections = detect_image(window, model)

            if GetWeightsType() == "tensorflow":
                detections = detect_image_tensorflow(window, tf_session)

            if GetWeightsType() == None:
                print("Error: No valid weights found.")
                print("Please supply a valid trained weights for detection.")
                sys.exit()

            if window_width != winW or window_height != winH:
                print("Non-square detection window detected. Window dimension: (" + str(window_width) + ", " + str(window_height) + ")")

            if detections is not None:
                if GetWeightsType() == "yolo" or GetWeightsType() == 'pytorch':
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

            # cv2.imshow("Window", image)
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

def GetWeightsType():
    if opt.weights_path.endswith(".weights"):
        return "yolo"
    if opt.weights_path.endswith(".pth"):
        return "pytorch"
    if opt.weights_path.endswith(".pb"):
        return "tensorflow"
    if opt.weights_path.endswith("h5"):
        return "keras"

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, help="path to model definition file")
    parser.add_argument("--weights_path", type=str, required=True, help="path to trained weights")
    parser.add_argument("--class_path", type=str, required=True, help="path to class label file")
    parser.add_argument("--image", type=str, required=True, help="the image to apply sliding windows on")
    parser.add_argument("--output", type=str, required=True, help="path to the detections output")
    parser.add_argument("--window_size", type=int, required=True, help="size of the sliding window")

    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
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
    # print("Image width: " + str(image_width) + " image height: " + str(image_height))

    classes = load_classes(opt.class_path)

    if GetWeightsType() == "yolo" or GetWeightsType() == "pytorch":

        from torch.utils.data import DataLoader
        from torchvision import datasets
        from torch.autograd import Variable

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        print("PyTorch model detected.")
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            print("Loaded the full weights with network architecture.")
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            print("Loaded only the trained weights.")
            model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))

        print("Weights: " + os.path.basename(opt.weights_path) + ".")
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
            sliding_windows([winW, winH])
            end_time = time.time()
            print("Time elapsed for YOLO/Pytorch detection: " + str(end_time - start_time) + "s.")


    elif GetWeightsType() == "tensorflow":
        import tensorflow as tf
        print("Tensorflow weights detected.")
        print("Loaded tensorflow weights: " + os.path.basename(opt.weights_path) + ".")

        [winW, winH] = [opt.window_size, opt.window_size]
        opt.x_stride = int(winW / 2)
        opt.y_stride = int(winH / 2)

        global_var.max_x = (image_width / opt.x_stride) - 1
        global_var.max_y = (image_height / opt.y_stride) - 1

        print("Running sliding windows on " + opt.image + ". Window dimension: [" + str(winW) + ", " + str(winH) + "]")

        start_time = time.time()

        with tf.gfile.FastGFile(opt.weights_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        )

        sess = tf.Session(config=config)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        sliding_windows([winW, winH], sess)
        sess.close()
        end_time = time.time()
        print("Time elapsed for Tensorflow detection: " + str(end_time - start_time) + "s.")

    else:
        print("Could not find a valid trained weights for detection. Please supply a valid weights")
        sys.exit()

    output_path = opt.output

    with open(os.path.join(output_path, 'detection.json')) as json_file:
        input_json = json.load(json_file)

    image_before_filter = imread(opt.image, plugin='pil')
    draw_bounding_boxes(input_json, image_before_filter)
    cv2.imwrite(os.path.join(output_path, "detection_before_filter.jpeg"), image_before_filter)

    input_json = SortDetections(input_json)

    iou_thres_range = [0.5]

    filtering_start = time.time()
    for iou_thres in iou_thres_range:
        # input_json = filter_bounding_boxes(input_json, iou_thres)
        input_json = filter_bounding_boxes_optimized(input_json, iou_thres)
    filtering_end = time.time()

    print("Bounding box filtering elapsed time: " + str(filtering_end - filtering_start))

    with open(os.path.join(output_path, "detection_filtered.json"), "w") as img_json:
        json.dump(input_json, img_json, indent=4)

    image = imread(opt.image, plugin='pil')
    draw_bounding_boxes(input_json, image)
    cv2.imwrite(os.path.join(output_path, "detection.jpeg"), image)

    image_circles = imread(opt.image, plugin='pil')
    draw_circles(input_json, image_circles)
    cv2.imwrite(os.path.join(output_path, "detection_circles.jpeg"), image_circles)
