import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


import sys
import os
import numpy as np
import caffe
import matplotlib.pyplot as plt

model_def = 'code/lab04/caffenet/deploy.prototxt'
model_weights = 'code/lab04/caffenet/bvlc_reference_caffenet.caffemodel'

#Net loading parameters changed in Python 3
net = caffe.Net(model_def, 1  ,weights=model_weights)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('code/lab04/caffenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', list(zip('BGR', mu)))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image('data/cat.jpg')
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

#Set to CPU mode as there's a bug in the deprecated layer the model's using
caffe.set_mode_cpu()


### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print('predicted class is:', output_prob.argmax())

# load ImageNet labels
labels_file = 'code/lab04/caffenet/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

print('output label:', labels[output_prob.argmax()])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print('probabilities and labels:')
print(list(zip(output_prob[top_inds], labels[top_inds])))


# for each layer, show the output shape
for layer_name, blob in net.blobs.items():
    print(layer_name + '\t' + str(blob.data.shape))

# for each layer, show the parameters
for layer_name, param in net.params.items():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

# A function to visualise blobs and parameters
def vis_square(data, figname):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data); plt.axis('off')
    plt.savefig(figname+".png")

# Visualising conv1 parameters
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1), "conv1_params")

# Visualising conv1 output
feat = net.blobs['conv1'].data[0]
vis_square(feat, "conv1_output")

feat = net.blobs['conv2'].data[0]
vis_square(feat, "conv2_output")

feat = net.blobs['conv3'].data[0]
vis_square(feat, "conv1_output")

feat = net.blobs['conv4'].data[0]
vis_square(feat, "conv4_output")

feat = net.blobs['conv5'].data[0]
vis_square(feat, "conv5_output")

# Visualising pool5 output
feat = net.blobs['pool5'].data[0]
vis_square(feat, "pool5_output")

# Plot histogram of fc6 output
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig("fc6_output.png")

# Plot output probability
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.savefig("prob_distribution.png")
