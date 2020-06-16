"""
apply_nms.py: Wrapper for nms.py
Authors : svp
"""

import numpy as np

'''
	nms.py: CPU implementation of non maximal supression modified from Ross's code.
	Authors : svp

	Modified from https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
	to accommodate a corner case which handles one box lying completely inside another.
'''


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        remove_index_1 = np.where(areas[i] == inter)
        remove_index_2 = np.where(areas[order[1:]] == inter)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr[remove_index_1] = 1.0
        ovr[remove_index_2] = 1.0
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


'''
    Extracts confidence map and box map from N (N=4 here)
	channel input.

    Parameters:
    -----------
    confidence_map - (list) list of confidences for N channels
    hmap - (list) list of box values for N channels

    Returns
    -------
    nms_conf_map - (HXW) single channel confidence score map 
	nms_conf_box - (HXW) single channel box map.
'''


def extract_conf_points(confidence_map, hmap):
    nms_conf_map = np.zeros_like(confidence_map[0])
    nms_conf_box = np.zeros_like(confidence_map[0])

    idx_1 = np.where(np.logical_and(confidence_map[0] > 0, confidence_map[1] <= 0))
    idx_2 = np.where(np.logical_and(confidence_map[0] <= 0, confidence_map[1] > 0))
    idx_common = np.where(np.logical_and(confidence_map[0] > 0, confidence_map[1] > 0))

    nms_conf_map[idx_1] = confidence_map[0][idx_1]
    nms_conf_map[idx_2] = confidence_map[1][idx_2]

    nms_conf_box[idx_1] = hmap[0][idx_1]
    nms_conf_box[idx_2] = hmap[1][idx_2]

    for ii in range(len(idx_common[0])):
        x, y = idx_common[0][ii], idx_common[1][ii]
        if confidence_map[0][x, y] > confidence_map[1][x, y]:
            nms_conf_map[x, y] = confidence_map[0][x, y]
            nms_conf_box[x, y] = hmap[0][x, y]
        else:
            nms_conf_map[x, y] = confidence_map[1][x, y]
            nms_conf_box[x, y] = hmap[1][x, y]

    assert (np.sum(nms_conf_map > 0) == len(idx_1[0]) + len(idx_2[0]) + len(idx_common[0]))

    return nms_conf_map, nms_conf_box


'''
    Wrapper function to perform NMS

    Parameters:
    -----------
    confidence_map - (list) list of confidences for N channels
    hmap - (list) list of box values for N channels
    wmap - (list) list of box values for N channels	
    dotmap_pred_downscale -(int) prediction scale
    thresh - (float) Threshold for NMS.

    Returns
    -------
    x, y - (list) list of x-coordinates and y-coordinates to keep
           after NMS.
    h, w - (list) list of height and width of the corresponding x, y 
            points.
    scores - (list) list of confidence for h and w at (x, y) point.

'''


def apply_nms(confidence_map, hmap, wmap, dotmap_pred_downscale=2, thresh=0.3):
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[0], confidence_map[1]], [hmap[0], hmap[1]])
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[2], nms_conf_map], [hmap[2], nms_conf_box])
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[3], nms_conf_map], [hmap[3], nms_conf_box])

    confidence_map = nms_conf_map
    hmap = nms_conf_box
    wmap = nms_conf_box

    confidence_map = np.squeeze(confidence_map)
    hmap = np.squeeze(hmap)
    wmap = np.squeeze(wmap)

    dets_idx = np.where(confidence_map > 0)

    y, x = dets_idx[-2], dets_idx[-1]
    h, w = hmap[dets_idx], wmap[dets_idx]
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    scores = confidence_map[dets_idx]

    dets = np.stack([np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores)], axis=1)
    # List of indices to keep
    keep = nms(dets, thresh)

    y, x = dets_idx[-2], dets_idx[-1]
    h, w = hmap[dets_idx], wmap[dets_idx]
    x = x[keep]
    y = y[keep]
    h = h[keep]
    w = w[keep]

    scores = scores[keep]
    return x, y, h, w, scores
