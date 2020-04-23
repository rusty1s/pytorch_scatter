#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
  sudo apt-get install software-properties-common --yes
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test --yes
  sudo apt update
  sudo apt install gcc-7 g++-7 --yes
  export CC=gcc-7
  export CXX=g++-7
fi

