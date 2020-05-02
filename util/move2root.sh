#!/bin/bash
find . -type f -name '*.jpg' |
    while read a; do
        ((c++))
        base="${a##*/}"
        mv "$a" "./${base%.jpg}_$(printf $c).jpg"
done
