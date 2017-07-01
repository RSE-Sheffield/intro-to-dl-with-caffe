import sys
import os
import numpy as np
import caffe

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
