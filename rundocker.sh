#!/bin/bash

nvidia-docker run --net=host -ti --rm -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/ubuntu:/home/developer  rses/dockercaffe
