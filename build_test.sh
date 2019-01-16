#!/bin/bash
clear
g++ -std=c++11 -DDLIB_JPEG_SUPPORT -DDLIB_PNG_SUPPORT -DDLIB_USE_BLAS -DDLIB_USE_LAPACK -DLAPACK_FORCE_UNDERSCORE -O3 -I.. ../dlib/all/source.cpp -I/opt/X11/include -L/usr/X11/lib -L/usr/X11R6/lib -lpthread -lX11 -ljpeg -lpng -llapack -lblas test.cpp -o ./build/test
