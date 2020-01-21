#!/bin/bash

set -xe

# Run the unittests for the ML libs
cd ..
python3 -m unittest discover -s analysis/tests -v
cd website

# Wait for backend database connection
/bin/bash wait-for-it.sh

## Needs a connection to a DB so migrations go here
python3 manage.py makemigrations website
python3 manage.py migrate
python3 manage.py startcelery

# Run website unittests
python3 manage.py test --noinput -v 2
