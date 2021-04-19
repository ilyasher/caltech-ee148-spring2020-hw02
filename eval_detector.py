import os
import json
import numpy as np

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    def area(box):
        dy = box[2]-box[0]+1
        dx = box[3]-box[1]+1
        if dy <= 0 or dx <= 0:
            return 0
        return dx * dy

    tl_row1, tl_col1, br_row1, br_col1 = tuple(box_1)
    tl_row2, tl_col2, br_row2, br_col2 = tuple(box_2)

    inter = 4 * [0]
    inter[0] = max(box_1[0], box_2[0])
    inter[1] = max(box_1[1], box_2[1])
    inter[2] = min(box_1[2], box_2[2])
    inter[3] = min(box_1[3], box_2[3])

    a1 = area(box_1)
    a2 = area(box_2)
    ai = area(inter)

    iou = ai / (a1 + a2 - ai)

    # print(a1, a2, ai, iou)

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''

    def is_gt_found(gt_box, pred):
        for pred_box in pred:
            iou = compute_iou(pred_box[:4], gt_box)
            if iou >= iou_thr:
                return True

        return False

    for pred_file, pred in preds.items():
        gt = gts[pred_file]

        found = 0

        # Filter out low-confidence predictions
        pred = [p for p in pred if p[4] >= conf_thr]

        # Take advantage of the fact that traffic lights
        # (ground truths) won't overlap

        for gt_box in gt:
            if is_gt_found(gt_box, pred):
                found += 1

        TP += found
        FN += len(gt) - found
        FP += len(pred) - found

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

def draw_pr_curve(preds_dict, gts):
    all_boxes = list(preds_dict.values())
    confidence_thrs = list()
    for preds in preds_dict.values():
        for box in preds:
            confidence_thrs.append(box[4])

    import random
    import matplotlib.pyplot as plt
    random.seed(23)
    confidence_thrs = random.choices(confidence_thrs, k=256)

    fig, ax = plt.subplots()

    ious = [0.25, 0.5, 0.75]
    precision = np.zeros((len(ious), len(confidence_thrs)))
    recall    = np.zeros((len(ious), len(confidence_thrs)))
    color     = np.zeros((len(ious), len(confidence_thrs)))

    for i, iou_thr in enumerate(ious):
        tp_train = np.zeros(len(confidence_thrs))
        fp_train = np.zeros(len(confidence_thrs))
        fn_train = np.zeros(len(confidence_thrs))
        for j, conf_thr in enumerate(confidence_thrs):
            tp_train[j], fp_train[j], fn_train[j] = compute_counts(preds_dict, gts, iou_thr=iou_thr, conf_thr=conf_thr)

        precision[i] = tp_train / (tp_train + fp_train)
        recall[i] = tp_train / (tp_train + fn_train)
        color[i] = iou_thr
        print(f"Progress: {i+1}/{len(ious)}")

    scatter = ax.scatter(recall.reshape(len(ious)*len(confidence_thrs)),
                        precision.reshape(len(ious)*len(confidence_thrs)),
                        c=color.reshape(len(ious)*len(confidence_thrs)),
                        cmap='viridis',
                        vmin=0,
                        label=f"IOU > {iou_thr}",
                        vmax=1)

    legend1 = ax.legend(*scatter.legend_elements(),
                        title="IoU Threshold")
    ax.add_artist(legend1)

    plt.title(f"P-R Curve for Traffic Light Detector")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

draw_pr_curve(preds_train, gts_train)
if done_tweaking:
    draw_pr_curve(preds_test, gts_test)
