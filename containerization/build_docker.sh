#!/bin/bash

git rev-parse --short HEAD > ./git.hash

TAG="deepmi/lit:dev"
DOCKERFILE="./containerization/Dockerfile"

function usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help        Print this message and exit"
  echo "  -t, --tag TAG     Specify the Docker tag (default: deepmi/lit:dev)"
  echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    -t|--tag)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: -t|--tag requires a value"
        usage
        exit 1
      fi
      TAG="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown argument $1"
      usage
      exit 1
      ;;
  esac
done

docker build . -t "$TAG" -f "$DOCKERFILE"

rm ./git.hash
