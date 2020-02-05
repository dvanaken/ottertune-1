#!/bin/bash

addrport="0.0.0.0:8000"

# Wait for backend connection
/bin/bash wait-for-it.sh

# Kill any existing celery processes
pkill -9 -f celery
rm -f *.pid

## Needs a connection to a DB so migrations go here
python3 manage.py makemigrations website
python3 manage.py migrate
python3 manage.py createuser admin $ADMIN_PASSWORD --superuser
python3 manage.py celery multi restart

echo ""
echo "-=------------------------------------------------------"
echo " Starting the web server on '$addrport'..."
echo "-=------------------------------------------------------"
python3 manage.py runserver "$addrport"
