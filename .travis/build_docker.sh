#!/bin/bash

# Exits if any command returns a non-zero return value
set -ex

cd $ROOT
echo "$DOCKER_PASSWD" | docker login -u "$DOCKER_USER" --password-stdin > /dev/null 2>&1

for tag in base web driver; do

    # Use current image in registry as build cache
    docker pull "${DOCKER_REPO}:${tag}" > /dev/null 2>&1

    docker image ls

    docker build \
      -f "docker/Dockerfile.${tag}" \
      -t "ottertune-${tag}" \
      --cache-from "${DOCKER_REPO}:${tag}" \
      --label "NAME=ottertune-${tag}" \
      --label "git_commit=$TRAVIS_COMMIT" \
      --build-arg DEBUG=true \
      --build-arg GIT_COMMIT=$TRAVIS_COMMIT \
      .

    docker image ls
done

for tag in driver web base; do docker rmi "${DOCKER_REPO}:${tag}"; done
docker image ls
