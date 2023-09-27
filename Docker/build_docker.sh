#!/bin/bash
#
# This is a script to build the docker
# This small script specifies the version of the docker,
# puts that same version in the tag of the docker image,
# and gives it as a build argument into the docker

# Define version
version='v1.0'

# Get directory of this script
docker_dir=`dirname $0`

# Build docker with version in tag
tag="unet-attention:${version}"
docker build --build-arg version_arg=${version} --tag=${tag} -f ${docker_dir}/dockerfile ${docker_dir}
