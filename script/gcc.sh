#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test --yes
  sudo apt install gcc-7 g++-7 --yes
  export CC=gcc-7
  export CXX=g++-7
fi

