#!/bin/bash

# Exits if any command returns a non-zero return value
set -e

for tag in base web driver 
do
    docker tag "ottertune-${tag}" "${DOCKER_REPO}:${tag}" > /dev/null 2>&1
done
