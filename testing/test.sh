#!/bin/sh

echo "Test Start"
for file in `\find programs -maxdepth 1 -type f`; do
    # TODO
    echo "[Start]" $file
    upprism $file
    code=$?
    if [ ${code} -eq 0 ]; then
      echo "[Success]" $file
    else
      exit 1
    fi

done
