import cv2
import torch
import numpy as np
from utils_nms import apply_nms


def compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE):

    BOX_SIZE_BINS = [1]
    g_idx = 0
    while len(BOX_SIZE_BINS) < NUM_BOXES_PER_SCALE * len(PRED_DOWNSCALE_FACTORS):
        gamma_idx = len(BOX_SIZE_BINS) // (len(GAMMA) - 1)
        box_size = BOX_SIZE_BINS[g_idx] + GAMMA[gamma_idx]
        BOX_SIZE_BINS.append(box_size)
        g_idx += 1

    BOX_SIZE_BINS_NPY = np.array(BOX_SIZE_BINS)
    BOXES = np.reshape(BOX_SIZE_BINS_NPY, (4, 3))
    BOXES = BOXES[::-1]

    return BOXES, BOX_SIZE_BINS


def upsample_single(input_, factor=2):
    channels = input_.size(1)
    indices = torch.nonzero(input_)
    indices_up = indices.clone()
    # Corner case!
    if indices_up.size(0) == 0:
        #return torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor).cuda()
        return torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor)
    indices_up[:, 2] *= factor
    indices_up[:, 3] *= factor

    #output = torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor).cuda()
    output = torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor)
    output[indices_up[:, 0], indices_up[:, 1], indices_up[:, 2], indices_up[:, 3]] = input_[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

    output[indices_up[:, 0], channels-1, indices_up[:, 2]+1, indices_up[:, 3]] = 1.0
    output[indices_up[:, 0], channels-1, indices_up[:, 2], indices_up[:, 3]+1] = 1.0
    output[indices_up[:, 0], channels-1, indices_up[:, 2]+1, indices_up[:, 3]+1] = 1.0

    # output_check = nn.functional.max_pool2d(output, kernel_size=2)

    return output


def get_upsample_output(model_output, output_downscale):
    upsample_max = int(np.log2(16 // output_downscale))
    upsample_pred = []
    for idx, out in enumerate(model_output):
        out = torch.nn.functional.softmax(out, dim=1)
        upsample_out = out
        for n in range(upsample_max - idx):
            upsample_out = upsample_single(upsample_out, factor=2)
        upsample_pred.append(upsample_out.cpu().data.numpy().squeeze(0))
    return upsample_pred


def box_NMS(predictions, nms_thresh, BOXES):
    Scores = []
    Boxes = []
    for k in range(len(BOXES)):
        scores = np.max(predictions[k], axis=0)
        boxes = np.argmax(predictions[k], axis=0)
        # index the boxes with BOXES to get h_map and w_map (both are the same for us)
        mask = (boxes < 3)  # removing Z
        boxes = (boxes + 1) * mask
        scores = (scores * mask)  # + 100 # added 100 since we take logsoftmax and it's negative!!

        boxes = (boxes == 1) * BOXES[k][0] + (boxes == 2) * BOXES[k][1] + (boxes == 3) * BOXES[k][2]
        Scores.append(scores)
        Boxes.append(boxes)

    x, y, h, w, scores = apply_nms(Scores, Boxes, Boxes, 0.5, thresh=nms_thresh)

    nms_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2]))  # since predictions[0] is of size 4 x H x W
    box_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2]))  # since predictions[0] is of size 4 x H x W
    for (xx, yy, hh) in zip(x, y, h):
        nms_out[yy, xx] = 1
        box_out[yy, xx] = hh

    assert (np.count_nonzero(nms_out) == len(x))

    return nms_out, box_out


def get_box_and_dot_maps(pred, nms_thresh, BOXES):
    assert (len(pred) == 4)
    # NMS on the multi-scale outputs
    nms_out, h = box_NMS(pred, nms_thresh, BOXES)
    return nms_out, h


def get_boxed_img(image, h_map, w_map, gt_pred_map, prediction_downscale, BOXES, BOX_SIZE_BINS,
                  thickness=1, multi_colours=False):
    if multi_colours:
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # colours for [1/8, 1/4, 1/2] scales

    if image.shape[2] != 3:
        boxed_img = image.astype(np.uint8).transpose((1, 2, 0)).copy()
    else:
        boxed_img = image.astype(np.uint8).copy()
    head_idx = np.where(gt_pred_map > 0)

    H, W = boxed_img.shape[:2]

    Y, X = head_idx[-2] , head_idx[-1]
    for y, x in zip(Y, X):

        h, w = h_map[y, x]*prediction_downscale, w_map[y, x]*prediction_downscale

        if multi_colours:
            selected_colour = colours[(BOX_SIZE_BINS.index(h // prediction_downscale)) // 3]
        else:
            selected_colour = (0, 255, 0)
        if h//2 in BOXES[3] or h//2 in BOXES[2]:
            t = 1
        else:
            t = thickness
        cv2.rectangle(boxed_img, (max(int(prediction_downscale * x - w / 2), 0), max(int(prediction_downscale * y - h / 2), 0)),
                      (min(int(prediction_downscale * x + w - w / 2), W), min(int(prediction_downscale * y + h - h / 2), H)), selected_colour, t)
    return boxed_img#.transpose((2, 0, 1))
