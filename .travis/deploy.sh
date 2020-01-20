#!/bin/bash

# Exits if any command returns a non-zero return value
set -e

for service in ${DOCKER_SERVICES}
do
    docker push "${DOCKER_REPO}:${service}"
done

