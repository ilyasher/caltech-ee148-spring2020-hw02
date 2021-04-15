import os
import numpy as np
import json
from PIL import Image

templates_dir = './templates/red-light'
template_files = sorted(os.listdir(templates_dir))
templates = [np.load(os.path.join(templates_dir, f)) for f in template_files if 'template' in f]

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
    (k_height, k_width, k_channels) = T.shape
    k_area = k_height * k_width
    if n_channels != k_channels:
        raise ValueError('number of channels does not match')

    if k_height % 2 == 0 or k_width % 2 == 0:
        raise ValueError('Dimensions of kernels should be odd')

    padded_I = np.zeros((n_rows+k_height-1, n_cols+k_width-1, n_channels))
    pad_h, pad_w = int((k_height - 1)/2), int((k_width - 1)/2)
    padded_I[pad_h:-pad_h, pad_w:-pad_w, :] = I

    h_height = int(n_rows / stride)
    h_width = int(n_cols / stride)
    heatmap = np.zeros((h_height, h_width))

    for hrow in range(h_height):
        for hcol in range(h_width):
            row = hrow * stride
            col = hcol * stride
            heatmap[hrow, hcol] = np.sum(padded_I[row:row+k_height, col:col+k_width, :] * T)

    heatmap = heatmap / k_area # Should also normalize at some point
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, stride):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    n_rows, n_cols = np.shape(heatmap)[:2]

    threshold = 100
    hits = heatmap > threshold

    # Simple recursive "island-finding" algorithm
    def recursive_grouper(i, j, bbox, hits):
        # Out of bounds
        if i < 0 or i >= n_rows or j < 0 or j >= n_cols:
            return False

        # Not a detection, or already visited
        if not hits[i, j]:
            return False

        print("hit!")

        # Improve bounding box
        bbox[0] = min(bbox[0], i)
        bbox[1] = min(bbox[1], j)
        bbox[2] = max(bbox[2], i)
        bbox[3] = max(bbox[3], j)

        # Improve confidence score
        bbox[4] += heatmap[i, j]

        # Mark this pixel as visited.
        hits[i, j] = False

        # Search the rest of the island.
        recursive_grouper(i-1, j, bbox, hits)
        recursive_grouper(i+1, j, bbox, hits)
        recursive_grouper(i, j-1, bbox, hits)
        recursive_grouper(i, j+1, bbox, hits)

        # We found a new detection
        return True

    for i in range(n_rows):
        for j in range(n_cols):
            bbox = [n_rows, n_cols, 0, 0, 0]
            if recursive_grouper(i, j, bbox, hits):
                # Scale confidence to [0, 1]
                confidence = 1 / (1 + 2.0**(-bbox[4])) # Sigmoid
                bbox[0] = (bbox[0] - 1) * stride
                bbox[1] = (bbox[1] - 1) * stride
                bbox[2] = (bbox[2] + 1) * stride
                bbox[3] = (bbox[3] + 1) * stride
                bbox[4] = confidence
                output.append(bbox)

    # Alternatively, no island-finding

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
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
    # template_height = 8
    # template_width = 6

    # You may use multiple stages and combine the results
    # T = np.random.random((template_height, template_width))

    output = list()
    for T in templates:
        # Ensure that T is odd-dimensional
        if T.shape[0] % 2 == 0:
            T = T[:-1, :, :]
        if T.shape[1] % 2 == 0:
            T = T[:, :-1, :]
        stride = int((min(T.shape[0], T.shape[1]) - 1) / 2)
        heatmap = compute_convolution(I, T, stride)
        output += predict_boxes(heatmap, stride)

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
done_tweaking = False

def make_outline(h, w):
    outline = np.zeros(shape=(h, w, 3))
    outline[:, :, 1] = 255
    outline[2:-2, 2:-2, 1] = 0
    return outline

'''
Make predictions on the training set.
'''
preds_train = {}
for i, filename in enumerate(file_names_train):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,filename))

    # convert to numpy array:
    I = np.asarray(I)

    print(i)
    preds = detect_red_light_mf(I)
    preds_train[filename] = preds

    # Visualizes predictions
    I = I.copy()
    for box in preds:
        print(filename, box)
        y0, x0, y1, x1, _ = tuple(box)
        cutout = I[y0:y1+1, x0:x1+1, :]
        cutout = np.maximum(cutout, make_outline(y1-y0+1, x1-x0+1))
        I[y0:y1+1, x0:x1+1, :] = cutout
    Image.fromarray(I).save(os.path.join(preds_path, filename))


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)


if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
