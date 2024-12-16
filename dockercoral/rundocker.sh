#!/bin/bash
xhost +local:docker
sudo docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb -v /home/egor/dockercoral/:/app/data -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1  --device=/dev/video0 coral-python:1 /bin/bash
