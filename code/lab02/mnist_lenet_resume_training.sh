#!/bin/bash

caffe train -solver=code/lab02/mnist_simple_solver.prototxt -snapshot=code/lab02/mnist_lenet_iter_5000.solverstate
