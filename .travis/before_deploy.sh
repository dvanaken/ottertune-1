#!/bin/bash

# Exits if any command returns a non-zero return value
set -e

for service in ${DOCKER_SERVICES}
do
    docker tag "ottertune-${service}" "${DOCKER_REPO}:${service}"
done

echo "$DOCKER_PASSWD" | docker login -u "$DOCKER_USER" --password-stdin
