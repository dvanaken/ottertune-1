#!/bin/bash

# Exits if any command returns a non-zero return value
set -ex

for tag in base web driver 
do
    docker tag "ottertune-${tag}" "${DOCKER_REPO}:${tag}" > /dev/null 2>&1
done

echo "$DOCKER_PASSWD" | docker login -u "$DOCKER_USER" --password-stdin > /dev/null 2>&1
