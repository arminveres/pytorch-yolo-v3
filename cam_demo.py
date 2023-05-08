import time
from numpy._typing import NDArray
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from util import write_results, load_classes
from darknet import Darknet
from preprocess import prep_image
import random
import argparse
import pickle as pkl


# def get_test_input(input_dim, CUDA):
#     img = cv2.imread("imgs/messi.jpg")
#     img = cv2.resize(img, (input_dim, input_dim))
#     img_ = img[:, :, ::-1].transpose((2, 0, 1))
#     img_ = img_[np.newaxis, :, :, :] / 255.0
#     img_ = torch.from_numpy(img_).float()
#     img_ = Variable(img_)
#     if CUDA:
#         img_ = img_.cuda()
#     return img_


def write_box(x: torch.tensor, img: NDArray) -> NDArray:
    """
    Adds bounding box to image
    Params:
        x: tensor
        img: image to be shown
    """
    start_point = x[1:3].numpy().astype(int)
    end_point = x[3:5].numpy().astype(int)

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    print(label)
    # color = random.choice(colors)
    color = [0, 0, 255]
    thickness = 2

    cv2.rectangle(img, start_point, end_point, color, thickness)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    end_point = start_point[0] + t_size[0] + 3, start_point[1] + t_size[1] + 4
    cv2.rectangle(img, start_point, end_point, color, -1)
    cv2.putText(
        img,
        label,
        (start_point[0], start_point[1] + t_size[1] + 4),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        [225, 255, 255],
        1,
    )
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description="YOLO v3 Cam Demo")
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object Confidence to filter predictions",
        default=0.25,
    )
    parser.add_argument(
        "--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4
    )
    parser.add_argument(
        "--reso",
        dest="reso",
        help="Input res. of the network. Increase to increase accuracy. Decrease to increase speed",
        default="160",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = "video.avi"

    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), "Cannot capture source"

    frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img, orig_im, dim = prep_image(frame, inp_dim)

        # im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            # im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(Variable(img), CUDA)
        output = write_results(
            output, confidence, num_classes, nms=True, nms_conf=nms_thesh
        )

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            continue

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

        # im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        classes = load_classes("data/coco.names")
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write_box(x, orig_im), output))

        cv2.imshow("frame", orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        frames += 1
        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
