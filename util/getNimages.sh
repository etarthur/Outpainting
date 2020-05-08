#!/bin/bash
N=164250;
for i in ./test/*.jpg; do
  [ "$((N--))" = 0 ] && break
  cp $i ./baseline_half/test_half
done
