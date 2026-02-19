#!/bin/bash

git rev-parse --short HEAD > ./git.hash

TAG="deepmi/lit:dev"
DOCKERFILE="./containerization/Dockerfile"

while [[ $# -gt 0 ]]; do
  case $1 in
    --experimental)
      DOCKERFILE="./containerization/Dockerfile_experimental"
      shift
      ;;
    -t|--tag)
      TAG="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

docker build . -t "$TAG" -f "$DOCKERFILE"

rm ./git.hash
