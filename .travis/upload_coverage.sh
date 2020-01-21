#!/bin/bash

flags=""

if [ "$#" -eq 1 ]; then
    flags="-F $1"
fi

codecov "$flags" \
  || (sleep 5 && codecov "$flags") \
  || (sleep 5 && codecov "$flags") \
  && echo "Codecov did not collect coverage reports"
