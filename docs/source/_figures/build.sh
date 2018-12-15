#!/bin/bash

files=(add sub mul div mean max min std)

for name in "${files[@]}"; do
  pdflatex "$name"
  pdf2svg "$name.pdf" "$name.svg"
done
