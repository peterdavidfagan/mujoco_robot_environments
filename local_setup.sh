#!/bin/bash

# following instalation steps: see https://ollama.ai/download/linux 
curl https://ollama.ai/install.sh | sh
ollama pull codellama # fetch codellama

# install python environment
poetry install

# set environment variables for docker display forwarding
export DOCKER_XAUTH=/tmp/.docker.xauth
touch $DOCKER_XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $DOCKER_XAUTH nmerge -
