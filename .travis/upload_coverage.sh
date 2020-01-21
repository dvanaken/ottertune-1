#!/bin/bash

set -x

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 LABEL FILE"
fi

label="$1"
file=`realpath --relative-to="$ROOT" "$2"`

for i in {1..3}; do
    codecov -F "$label" -f "$file" && break || sleep 5
done

[ "$i" -eq 3 ] && echo "Codecov did not collect coverage reports" || 0
