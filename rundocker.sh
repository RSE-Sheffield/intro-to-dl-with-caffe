#!/bin/bash

nvidia-docker run -ti --rm -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/ubuntu:/home/ubuntu  rses/dockercaffe
