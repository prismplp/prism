#!/bin/sh

echo "Test Start"
for file in `\find programs -maxdepth 1 -type f`; do
    # TODO
    echo $file
    upprism $file
done