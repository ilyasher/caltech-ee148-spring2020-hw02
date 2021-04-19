import os
import numpy as np
import json
from PIL import Image

import sys

DATA_MEAN = 90
DATA_STD  = 65

def resize(arr, w, h):
    shape = list(arr.shape)
    shape[:2] = [h, w]
    ret = np.zeros(shape)
    old_h, old_w = arr.shape[:2]
    for x in range(w):
        for y in range(h):
            old_x = int(x / w * old_w)
            old_y = int(y / h * old_h)
            ret[y][x] = arr[old_y][old_x]
    return ret

weak=True

templates_dir = './templates/red-light'
template_files = sorted([f for f in os.listdir(templates_dir) if 'template' in f])
# template_files = [template_files[int(i)] for i in sys.argv[1:]]
if weak:
    template_files = [template_files[int(i)] for i in [1, 2, 4, 5]]

templates = list()
for f in template_files:
    template = np.load(os.path.join(templates_dir, f))
    templates.append(template)
    # templates.append(resize(template, int(template.shape[1] * 1.5), int(template.shape[0] * 1.5)))
    # templates.append(resize(template, int(template.shape[1] * 0.7), int(template.shape[0] * 0.7)))
print(f"Using {len(templates)} template{'' if len(templates) == 1 else 's'}: {template_files}")

def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''

    # To make things simple, let's fix the kernel dimensions to be odd.
    (k_height, k_width, k_channels) = T.shape
    k_area = k_height * k_width
    if n_channels != k_channels:
        raise ValueError('number of channels does not match')

    if k_height % 2 == 0 or k_width % 2 == 0:
        raise ValueError('Dimensions of kernels should be odd')

    # Pad the image with a border to make it easier to compute the dot products.
    padded_I = np.zeros((n_rows+k_height-1, n_cols+k_width-1, n_channels))
    pad_h, pad_w = int((k_height - 1)/2), int((k_width - 1)/2)
    padded_I[pad_h:-pad_h, pad_w:-pad_w, :] = I

    # Create the heatmap which we will return.
    heatmap = np.zeros((n_rows, n_cols))

    for row in range(0, n_rows, stride):
        for col in range(0, n_cols, stride):

            # The pixel value in the heatmap is the dot product of the
            # image section and the kernel.
            val = np.sum(padded_I[row:row+k_height, col:col+k_width, :] * T)
            heatmap[row-pad_h:row+pad_h, col-pad_w:col+pad_w] = np.maximum(heatmap[row-pad_h:row+pad_h, col-pad_w:col+pad_w], val)

    # We don't want the dot products to change too
    # much with the kernel size.
    heatmap = heatmap / (k_area ** 1)

    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    n_rows, n_cols = np.shape(heatmap)[:2]

    threshold = 2.5

    hits = heatmap > threshold

    def grouper(i, j, bbox, hits):
        q = [(i, j)]
        seen = set()

        if not hits[i, j]:
            return False

        while q:
            i, j = q.pop(0)
            if (i, j) in seen:
                continue
            seen.add((i, j))

            if i < 0 or i >= n_rows or j < 0 or j >= n_cols:
                continue

            # Not a detection
            if not hits[i, j]:
                continue

            hits[i, j] = False

            # Improve bounding box
            bbox[0] = min(bbox[0], i)
            bbox[1] = min(bbox[1], j)
            bbox[2] = max(bbox[2], i)
            bbox[3] = max(bbox[3], j)

            # Improve confidence score
            # bbox[4] += heatmap[i, j]
            bbox[4] = max(bbox[4], heatmap[i, j])

            # Search the rest of the island.
            for new in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if new not in q and new not in seen:
                    q.append(new)

        return True

    for i in range(n_rows):
        for j in range(n_cols):
            big = max(n_rows, n_cols)
            bbox = [big, big, 0, 0, 0]
            if grouper(i, j, bbox, hits):
                # Scale confidence to [0, 1]
                confidence = 1 / (1 + 2.0**(-bbox[4])) # Sigmoid
                print("Detection size ({}, {})\t Confidence: {}".format(bbox[3] - bbox[1], bbox[2] - bbox[0], confidence))
                bbox[4] = confidence
                output.append(bbox)

    '''
    END YOUR CODE
    '''

    return output

heatmaps_path = '../data/hw02_heatmaps'

def detect_red_light_mf(I, filename=None):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    I = I.astype(np.float32)

    # Normalize image
    I = (I - np.mean(I, axis=(0, 1))) / np.std(I, axis=(0, 1))

    heatmap = np.zeros(I.shape[:2])

    for T in templates:
        # Ensure that T is odd-dimensional
        if T.shape[0] % 2 == 0:
            T = T[:-1, :, :]
        if T.shape[1] % 2 == 0:
            T = T[:, :-1, :]
        stride = max(int((min(T.shape[0], T.shape[1]) - 1) / 5), 2)

        conv = compute_convolution(I, T, stride)
        heatmap = np.maximum(heatmap, conv)

    if filename:
        Image.fromarray(np.minimum(255, heatmap * 50).astype(np.uint8)).save(os.path.join(heatmaps_path, filename))

    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/red-lights'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

def make_outline(h, w):
    outline = np.zeros(shape=(h, w, 3))
    outline[:, :, 1] = 255
    outline[2:-2, 2:-2, 1] = 0
    return outline

'''
Make predictions on the training set.
'''
preds_train = {}
for i, filename in enumerate(sorted(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,filename))

    # convert to numpy array:
    I = np.asarray(I)

    print(f"Processing Image {i+1}/{len(file_names_train)}, name: {filename}")
    preds = detect_red_light_mf(I, filename=filename)
    preds_train[filename] = preds

    # Visualizes predictions
    I = I.copy()
    for box in preds:
        y0, x0, y1, x1, _ = tuple(box)
        cutout = I[y0:y1+1, x0:x1+1, :]
        cutout = np.maximum(cutout, make_outline(cutout.shape[0], cutout.shape[1]))
        I[y0:y1+1, x0:x1+1, :] = cutout
    Image.fromarray(I).save(os.path.join(preds_path, filename))


# save preds (overwrites any previous predictions!)
# TODO: uncomment
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)


if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i, filename in enumerate(sorted(file_names_test)):

        print(f"Processing Test Image {i+1}/{len(file_names_test)}, name: {filename}")

        # read image using PIL:
        I = Image.open(os.path.join(data_path,filename))

        # convert to numpy array:
        I = np.asarray(I)

        preds = detect_red_light_mf(I, filename)
        preds_test[filename] = preds

        I = I.copy()
        for box in preds:
            y0, x0, y1, x1, _ = tuple(box)
            cutout = I[y0:y1+1, x0:x1+1, :]
            cutout = np.maximum(cutout, make_outline(cutout.shape[0], cutout.shape[1]))
            I[y0:y1+1, x0:x1+1, :] = cutout
        Image.fromarray(I).save(os.path.join(os.path.join(preds_path, 'test'), filename))


    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
