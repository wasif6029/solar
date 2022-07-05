# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
import numpy as np
import pandas as pd
from tkinter import *
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch
from pathlib import Path
import sys
import math
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

centers = []
time = []
timeCount = 0

frame_height = 480
frame_width = 480

left_part = True
saving_directory = ''
final_video_dir = ''

area_points = []
tmp_area = 0


area = [420, 550, 624, 700, 783, 896, 1044, 1292, 1628, 1968, 2491, 3410, 4672, 6468, 10476, 18460, 44421]
z_depth = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]


x = []
y = []
z = []


def assumptionOfDepth(value_area):
    res = 0.0
    for i in range(len(area)):
        if value_area == area[i]:
            res = z_depth[i]
        elif value_area < area[i]:
            if i == 0:
                res = (area[i] / value_area) * z_depth[i]
            else:
                res = z_depth[i - 1] - (((value_area - area[i - 1]) *
                                        (z_depth[i - 1] - z_depth[i])) / (area[i] - area[i - 1]))
        else:
            if i == len(area) - 1:
                res = (area[i] / value_area) * z_depth[i]
            else:
                continue
    res = round(res, 2)

    return res


def assumptionZ(value_area):
    for i in range(len(area) - 1):
        if i == len(area) - 1:
            return int(area[i] / value_area * z_depth[i])
        elif value_area >= area[i] and value_area <= area[i + 1]:
            return (int((z_depth[i] + z_depth[i + 1]) / 2))


def assumptionZNew(value_area):
    for i in range(len(area) - 1):
        if i == len(area) - 1:
            return z_depth[i] * (1 - (area[i] / value_area))
        elif value_area >= area[i] and value_area <= area[i + 1]:
            return z_depth[i] - (z_depth[i] - z_depth[i + 1]) * ((value_area - area[i]) / (area[i + 1] - area[i]))
            # return (int((z_depth[i] + z_depth[i + 1]) / 2))
        elif value_area < area[i] and i == 0:
            return z_depth[i] * (1 + (value_area / area[i]))


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(frame_height, frame_width),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=True,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        left=False
):

    global left_part
    left_part = left

    global frame_height
    global final_video_dir
    global saving_directory

    global timeCount

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    saving_directory = save_dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # print(imgsz)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                newName = 'Wasif'
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # s += f"{n} {newName}{'s' * (n > 1)}, "  # add to string

                checks = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image

                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())

                        area = abs(x1 - x2) * abs(y1 - y2)

                        X = math.floor((x1 + x2) / 2)
                        YY = math.floor((y1 + y2) / 2)
                        Y = math.floor(frame_height - YY)
                        # Z = assumptionZ(area)
                        Z = assumptionZNew(area)
                        # Z = assumptionOfDepth(area)

                        x.append(X)
                        y.append(Y)
                        z.append(Z)
                        time.append(timeCount)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break   

                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{X, Y, Z}')
                        # names[c] if hide_conf else f'{names[c]} {X, Y, Z} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        area_points.append(int(area))

                        centers.append([math.floor((x1 + x2) / 2), math.floor((y1 + y2) / 2)])
                        checks = 0
                        confidence_score = conf
                        class_index = cls
                        object_name = names[int(cls)]

                        print(object_name, ' #', class_index, ' found at (', math.floor((
                            x1 + x2) / 2), math.floor((y1 + y2) / 2), ')point')

                    else:
                        checks = checks + 1
                        print(checks)
                        centers.append([None, None])

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frame_height = h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        final_video_dir = save_path
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        timeCount += 1

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(imgsz)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--left', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    # return centers


if __name__ == "__main__":

    print("In detect")
    opt = parse_opt()

    try:
        vcap = cv2.VideoCapture(opt.source)
        frame_width = vcap. get(cv2. CAP_PROP_FRAME_WIDTH)
        frame_height = vcap. get(cv2. CAP_PROP_FRAME_HEIGHT)
    except:
        frame_height, frame_width, c = cv2.imread(opt.source).shape

    tmp_area = 0

    main(opt)

    prev = -1
    results = []
    area_points.sort()
    for i in range(len(area_points)):
        if area_points[i] != prev:
            results.append(area_points[i])
            prev = area_points[i]

    nameOfCoordinates = '\\centers.txt'

    if left_part:
        nameOfCoordinates = '\\left_centers.txt'
    else:
        nameOfCoordinates = '\\right_centers.txt'

    # cap = cv2.VideoCapture(str(saving_directory)+"//")

    f = open(str(saving_directory) + nameOfCoordinates, 'w')

    for i in range(len(x)):
        f.writelines(str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + '\n')

    new_final_video_dir = final_video_dir.replace(".mp4", "_traced.mp4")

    cap = cv2.VideoCapture(final_video_dir)

    # output = cv2.VideoWriter(new_final_video_dir, cv2.VideoWriter_fourcc(*'MPEG'),
    #                          cv2.CAP_PROP_FPS, (int(frame_height), int(frame_width)))
    output = cv2.VideoWriter(new_final_video_dir, -1, cv2.CAP_PROP_FPS, (int(cap.get(3)), int(cap.get(4))))

    x_data = np.array(x)
    y_data = np.array(y)

    def model_f(x, a, b, c):
        return a * x**2 + b * x + c

    popt, pcov = curve_fit(model_f, x_data, y_data, p0=[3, 2, -16])

    a_opt, b_opt, c_opt = popt
    x_model = np.linspace(min(x_data), max(x_data), 100)
    y_model = model_f(x_model, a_opt, b_opt, c_opt)

    a_opt, b_opt, c_opt = popt
    x_model = np.linspace(min(x_data), max(x_data) + 300, 300)
    y_model = model_f(x_model, a_opt, b_opt, c_opt)

    while(True):
        ret, frame = cap.read()
        if(ret):
            for i in range(len(x_model)):
                # adding filled rectangle on each frame
                cv2.circle(frame, (int(x_model[i]), int(frame_height - y_model[i])), 2,
                           (0, 255, 0), -1)

            # writing the new frame in output
            output.write(frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        else:
            break

    cv2.destroyAllWindows()
    output.release()
    cap.release()
