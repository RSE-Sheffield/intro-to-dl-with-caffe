#!/usr/bin/env python
import numpy as np
import caffe
import json
import random


model_path = "code/lab05/text_gen_deploy.prototxt"
weights_path = "text_gen_iter_6000.caffemodel"

#Net loading parameters changed in Python 3
net = caffe.Net(model_path, 1, weights=weights_path)

caffe.set_mode_gpu()
caffe.set_device(0)

#Gets the json char index map
dict_input_file = "wonderland_dict.json"
uchar_to_int = json.loads(open(dict_input_file).read())

#Convert unicode keys to bytestring
char_to_int = {}
for c , i in uchar_to_int.items():
	char_to_int[c.encode('utf-8')] = i

#And make a reverse char lookup map
int_to_char = dict((i, c) for c, i in char_to_int.items())

num_vocab = len(int_to_char)
print "Total Vocab: ", num_vocab


#Predict sequence
seq_length = 50
no_predict = 1000 #Number of characters to generate

#We'll choose a random excerpt from the data
test_file = "data/looking_glass.txt"
raw_text = open(test_file,"r").read().lower()
raw_text_length = len(raw_text)
seed_start = random.randint(0,raw_text_length - seq_length)
seed_text = raw_text[seed_start:seed_start+seq_length]

#Replace any charcters not used in our training data with a blank space
filtered_text = ""
for i, c in enumerate(seed_text):
	if c not in char_to_int:
		filtered_text += " "
	else:
		filtered_text += c
seed_text = filtered_text

#Gets the input blobs
input_blob = net.blobs['input_sequence']
cont_blob = net.blobs['cont_sequence']


#Create ndarray for filling
input_np = np.zeros( (seq_length, 1), dtype="float32")
cont_np = np.zeros( (seq_length,1) , dtype="float32")
cont_np.fill(1) #Continuity is always 1

input_queue= []

#Get seed to fill the string
for c in seed_text:
	input_queue.append(c)

print "Seeding with text: \n", "".join(input_queue)

result = ""

#Generate some text
for i in range(no_predict):

	#Fill numpy arrays
	for j in range(seq_length):
		input_np[j,0] = char_to_int[input_queue[j]]

	#Fill the data blob
	input_blob.data[...] = input_np
	cont_blob.data[...] = cont_np
	output = net.forward()
	output_prob = output['probs']

	#Gets all the predicted characters and its confidence
	out_seq = []
	out_conf_seq = []
	for p in output_prob:
		#Gets the index with maximum probability
		out_max_index = p[0].argmax()
		#Get actual charcter
		predicted_char = int_to_char[out_max_index]
		#Get confidence of the prediction
		confidence = p[0,out_max_index]

		#Adds to the array
		out_seq.append(predicted_char)
		out_conf_seq.append(confidence)


	next_char = out_seq[-1]
	next_confidence = out_conf_seq[-1]


	#Print the result of this prediction
	print "Prediction no.: ", str(i)
	print "Input sequence: \n", input_queue
	print "Output sequence: \n", out_seq
	print "Output confidence: \n", out_conf_seq
	print "Next char prediction: ", repr(next_char), " confidence: ", str(next_confidence)

	#Add to the input list and pop the first character
	input_queue.append(next_char)
	popped_char = input_queue.pop(0)

	#Add text to the final string
	result += popped_char

#Fill the rest text with the remaining prediction
for c in input_queue:
	result += c

print "Seed text:"
print seed_text
print "Final output:"
print result
