#!/usr/bin/env python
import h5py
import numpy as np
import json

#Load  text and covert to lowercase
filename = "data/wonderland.txt"
dict_output_file = "wonderland_dict.json"
hdf_output_file = "wonderland.hdf5"
hdf_list_file = "wonderland_hdf5_list.txt"

raw_text = open(filename).read().lower()
raw_text = "".join(i for i in raw_text if ord(i)<128) #< Remove all non-ascii characters
print "Raw text length: ",len(raw_text)


total_length = int(50000)
num_streams = int(250)
stream_length = int(total_length/num_streams)

# add +1 to text so we have a predict the next char from the last
clipped_text = raw_text[0:total_length+1]


# create mapping of unique chars to integers
chars = sorted(list(set(clipped_text)))
char_to_int = {}
for i, c in enumerate(chars):
    print c, " ", i
    char_to_int[c] = i

print char_to_int

with open(dict_output_file, "w") as f:
    f.write(json.dumps(char_to_int))
    f.close()


# summarize the loaded data
n_chars = len(clipped_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
print "Text length: ", total_length , " stream length: ", stream_length, " numstreams: ", num_streams

input_data = []
cont_data = []
target_data = []

for i in range(num_streams):
    input_stream = []
    cont_stream = []
    target_stream = []
    for j in range(stream_length):
        text_index = i*stream_length + j
        char_in = clipped_text[text_index]
        char_target = clipped_text[text_index+ 1]
        input_stream.append(char_to_int[char_in])
        cont_stream.append(1)
        target_stream.append(char_to_int[char_target])
    input_data.append(input_stream)
    cont_data.append(cont_stream)
    target_data.append(target_stream)



input_np = np.array(input_data, dtype='float32')
input_np = np.transpose(input_np, (1, 0))

cont_np = np.array(cont_data, dtype='uint8')
cont_np = np.transpose(cont_np, (1,0))

target_np = np.array(target_data, dtype='float32')
target_np = np.transpose(target_np, (1, 0))


print "Input data shape: ", input_np.shape
print "Cont data shape: ", cont_np.shape
print "Target data shape: ", target_np.shape

#Create hdf5 file
with h5py.File(hdf_output_file, "w") as f:
    #Create dataset
    f.create_dataset("input_sequence", data=input_np)
    f.create_dataset("cont_sequence", data=cont_np)
    f.create_dataset("target_sequence", data=target_np)
    f.close()


#Create text file that has a path to hdf5
with open(hdf_list_file, "w") as f:
    f.write(hdf_output_file)
    f.close()


print "Dataset created"

#For completeness, here is how we would open and read a hdf5 file
# with h5py.File(hdf_output_file, "r") as f:
#     print f["input_sequence"].shape
#     print f["cont_sequence"].shape
#     print f["target_sequence"].shape
