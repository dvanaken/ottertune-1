#!/bin/bash

if [ -z "$BACKEND" ]
then
    echo " >> ERROR: Variable 'BACKEND' must be set." >&2
    exit 1
fi

DEBUG="${DEBUG:-true}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"
DB_NAME="${DB_NAME:-ottertune}"
DB_PASSWORD="${DB_PASSWORD:-ottertune}"

if [ "$BACKEND" = "mysql" ]; then
    DB_USER="${DB_USER:-root}"
    DB_INTERNAL_PORT=3306
    DB_IMAGE="mysql:5.6"
else
    DB_USER="${DB_USER:-postgres}"
    DB_INTERNAL_PORT=5432
    DB_IMAGE="postgres:9.6"
fi

DB_PORT="${DB_PORT:-$DB_INTERNAL_PORT}"
WEB_ENTRYPOINT="${WEB_ENTRYPOINT:-./start.sh}"

file="docker-compose.$BACKEND.yml"
driver_image="ottertune-driver"
driver_env="# env_file: N/A"
quiet=false

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [-f FILE] [--dev-driver]"
      echo ""
      echo "Options:"
      echo "  -h, --help        Display help"
      echo "  -f, --file FILE   Write output to FILE (default: docker-compose.$BACKEND.yml)"
      echo "      --dev-driver  Use the development driver image"
      echo "  -q, --quiet       Only display the output file"
      exit 0
      ;;
    -f|--file)
      file="$2"
      shift 2
      ;;
    --dev-driver)
      driver_image="${driver_image}-internal"
      driver_env="env_file: local_driver.env"
      shift 1
      ;;
    -q|--quiet)
      quiet=true
      shift 1
      ;;
    --) # end arg parsing
      shift
      break
      ;;
    *|-*|--*=) unsupported opts
      echo " >> ERROR: Unsupported option '$1'" >&2
      exit 1
      ;;
  esac
done

cat > $file <<- EOM
version: "3"
services:

    web:
        image: ottertune-web
        container_name: web
        expose:
          - "8000"
        ports:
          - "8000:8000"
        links:
          - backend
          - rabbitmq
        depends_on:
          - backend
          - rabbitmq
        environment:
          DEBUG: '$DEBUG'
          ADMIN_PASSWORD: '$ADMIN_PASSWORD'
          BACKEND: '$BACKEND'
          DB_NAME: '$DB_NAME'
          DB_USER: '$DB_USER'
          DB_PASSWORD: '$DB_PASSWORD'
          DB_HOST: 'backend'
          DB_PORT: '$DB_INTERNAL_PORT'
          MAX_DB_CONN_ATTEMPTS: 30
          RABBITMQ_HOST: 'rabbitmq'
        working_dir: /app/website
        entrypoint: $WEB_ENTRYPOINT
        labels:
          NAME: "ottertune-web"
        networks:
          - ottertune-net

    rabbitmq:
        image: "rabbitmq:3-management"
        container_name: rabbitmq
        restart: always
        hostname: "rabbitmq"
        environment:
           RABBITMQ_DEFAULT_USER: "guest"
           RABBITMQ_DEFAULT_PASS: "guest"
           RABBITMQ_DEFAULT_VHOST: "/"
        expose:
           - "15672"
           - "5672"
        ports:
           - "15673:15672"
           - "5673:5672"
        labels:
           NAME: "rabbitmq"
        networks:
          - ottertune-net

    driver:
        image: $driver_image
        container_name: driver
        depends_on:
          - web
        environment:
          DEBUG: '$DEBUG'
        $driver_env
        labels:
          NAME: "ottertune-driver"
        networks:
          - ottertune-net

EOM

cat >> $file <<- EOM
    backend:
        image: $DB_IMAGE
        container_name: backend
        restart: always
        environment:
EOM

if [ "$BACKEND" = "mysql" ]; then
cat >> $file <<- EOM
          MYSQL_USER: '$DB_USER'
          MYSQL_ROOT_PASSWORD: '$DB_PASSWORD'
          MYSQL_PASSWORD: '$DB_PASSWORD'
          MYSQL_DATABASE: '$DB_NAME'
EOM
else
cat >> $file <<- EOM
          POSTGRES_PASSWORD: '$DB_PASSWORD'
          POSTGRES_USER: '$DB_USER'
          POSTGRES_DB: '$DB_NAME'
EOM
fi

cat >> $file <<- EOM
        expose:
          - "$DB_INTERNAL_PORT"
        ports:
          - "$DB_PORT:$DB_INTERNAL_PORT"
        labels:
          NAME: "ottertune-backend"
        networks:
          - ottertune-net

networks:
   ottertune-net:
      driver: bridge
EOM

if [ "$quiet" = true ]; then
    echo "$file"
else
    echo "Saved docker-compose file to '$file'."
fi

