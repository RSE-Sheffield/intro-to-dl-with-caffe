#!/usr/bin/env python
import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks

def get_header_field(file):
    header_fields = []
    for line in file:
        line = line.strip()
        if line[0] == '#':
            line = line[1:] #Remove the hash and get the fields
            fields = line.split()
            for f in fields:
                header_fields.append(f.strip())

            break #Found our header

    if len(header_fields) < 1:
        print("No headers found")
        sys.exit(1)

    return header_fields

def check_file(log_path):
    if not os.path.isfile(log_path):
        print("%s is not a valid file")
        sys.exit(1)

def plot_chart(log_path, xaxis_name, yaxis_name, img_save_path):
    check_file(log_path)

    #Axis ids and a 2xN array for data
    xaxis_id = 0
    yaxis_id = 0
    data = [[], []]

    #Reading data file
    with open(log_path) as file:

        #Find headers first and get index
        header_fields = get_header_field(file)
        for i in range(len(header_fields)):
            if xaxis_name.lower() == header_fields[i].lower():
                xaxis_id = i
            if yaxis_name.lower() == header_fields[i].lower():
                yaxis_id = i

        for line in file:
            line = line.strip()
            if line[0] != '#':
                fields = line.split()
                data[0].append(float(fields[xaxis_id].strip()))
                data[1].append(float(fields[yaxis_id].strip()))


    print("xaxis %i yaxis %i" % (xaxis_id, yaxis_id))
    plt.plot(data[0], data[1])
    plt.title("%s vs %s" % (xaxis_name, yaxis_name))
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.xlim(0, max(data[0]))
    plt.ylim(0, max(data[1]))
    if img_save_path is not None:
        plt.savefig(img_save_path)
    plt.show()

def print_help():
    print("#Use as follows: \n #To get header fieds \n plot_log.py logfilePath \n #To plot the graph  \n plot_log.py logfilePath XAxisName YAxisName [imageSavePath.png] ")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_help()
    elif len(sys.argv) == 2:
        log_path = sys.argv[1]
        check_file(log_path)
        with open(log_path) as file:
            header_fields = get_header_field(file)
            print("Available headers: ")
            for h in header_fields:
                print(" %s " % h)

    else:
        log_path = sys.argv[1]
        xaxis_name = sys.argv[2]
        yaxis_name = sys.argv[3]
        img_save_path = None
        if len(sys.argv) > 4:
            img_save_path = sys.argv[4]
            if not img_save_path.lower().endswith(".png"):
                print("Image save path must be a .png")
        ## plot_chart accpets multiple path_to_logs
        plot_chart(log_path, xaxis_name, yaxis_name, img_save_path)
