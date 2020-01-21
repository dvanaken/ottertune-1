#!/bin/bash

# Exits if any command returns a non-zero return value
set -e

for tag in base web driver
do
    echo -n "Pushing image 'ottertune-$tag' to the registry... "
    docker push "${DOCKER_REPO}:${tag}" > /dev/null 2>&1
    echo "Done."
done

