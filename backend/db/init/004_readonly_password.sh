#!/bin/sh
# 用环境变量设置只读账号密码
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -c "ALTER ROLE text2sql_ro PASSWORD '${TEXT2SQL_RO_PASSWORD:-readonly}';"
