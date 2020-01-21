#!/bin/bash

set -ex

ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"

cd "$WEB"
sed -i  "s|\('celery', 'db.*$\)|'console', \1|" website/settings/common.py
cp "$ROOT/docker/credentials.py" website/settings
cat website/settings/credentials.py
python manage.py makemigrations
python manage.py migrate
python manage.py startcelery
python manage.py createuser admin "$ADMIN_PASSWORD" --superuser
