#!/bin/bash
xhost +local:docker 2>/dev/null

docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    --network host \
    --name neuroticks \
    neuroticks:latest

xhost -local:docker 2>/dev/null
