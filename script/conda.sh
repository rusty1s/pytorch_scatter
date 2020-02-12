#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b
  PATH=/home/travis/miniconda/bin:${PATH}
  conda update --yes conda
fi


