# --------------------------------------------------------
# Copyright (c) Zhejiang University
# Licensed under The MIT License [see LICENSE for details]
# Written by Fashuai Li
# --------------------------------------------------------

import numpy as np

#def nms(dets, thresh):
#    x1 = dets[:, 0]
#    y1 = dets[:, 1]
#    x2 = dets[:, 2]
#    y2 = dets[:, 3]
#    scores = dets[:, 4]
#
#    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#    order = scores.argsort()[::-1]
#
#    keep = []
#    while order.size > 0:
#        i = order[0]
#        keep.append(i)
#        xx1 = np.maximum(x1[i], x1[order[1:]])
#        yy1 = np.maximum(y1[i], y1[order[1:]])
#        xx2 = np.minimum(x2[i], x2[order[1:]])
#        yy2 = np.minimum(y2[i], y2[order[1:]])
#
#        w = np.maximum(0.0, xx2 - xx1 + 1)
#        h = np.maximum(0.0, yy2 - yy1 + 1)
#        inter = w * h
#        ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#        inds = np.where(ovr <= thresh)[0]
#        order = order[inds + 1]
#
#    return keep


def nms(det_pts, scores, thresh):

     det_pts = np.array(det_pts)
     scores = scores.detach().cpu().numpy()

     x = det_pts[:, 0]
     y = det_pts[:, 1]

     order = scores.argsort()[::-1]

     keep = []
     while order.size > 0:
         i = order[0]
         keep.append(i)
         dist = (x[i]-x[order[1:]])*(x[i]-x[order[1:]]) + (y[i]-y[order[1:]])*(y[i]-y[order[1:]])
         inds = np.where(dist >= thresh*thresh)[0]
         order = order[inds + 1]

     return det_pts[keep]


def nms_array(det_pts, scores, thresh):
    x = det_pts[:, 0]
    y = det_pts[:, 1]

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        dist = (x[i] - x[order[1:]]) * (x[i] - x[order[1:]]) + (y[i] - y[order[1:]]) * (y[i] - y[order[1:]])
        inds = np.where(dist >= thresh * thresh)[0]
        order = order[inds + 1]

    return det_pts[keep]


def NMS_tst(det_pts, thresh):
     #det_pts = np.mat(det_pts)

     x = det_pts[:, 0]
     y = det_pts[:, 1]
     scores = det_pts[:, 2]

     order = scores.argsort()[::-1]
     #order = torch.sort(scores)

     keep = []
     while order.size > 0:
         i = order[0]
         keep.append(i)
         dist = (x[i]-x[order[1:]])*(x[i]-x[order[1:]]) + (y[i]-y[order[1:]])*(y[i]-y[order[1:]])
         inds = np.where(dist >= thresh*thresh)[0]
         order = order[inds + 1]

     return keep


if __name__ == "__main__":
    det_pts = np.array([[10, 10, 0.99],
                        [10.1, 10.1, 0.98],
                        [10.3, 10.4, 0.97],
                        [10.5, 10.5, 0.96],
                        [20, 20, 0.95],
                        [20.1, 20.1, 0.94],
                        [20.2, 20.2, 0.93],
                        [20.5, 20.5, 0.92],
                        [30, 30, 0.91],
                        [30.1, 30.1, 0.90],
                        [30.2, 30.2, 0.89],
                        [30.5, 30.5, 0.88]])

    threshold = 5.0

    keep_pts = NMS_tst(det_pts, threshold)

    print(keep_pts)

    print(det_pts[keep_pts])