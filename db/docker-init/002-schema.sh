#!/usr/bin/env bash
set -Eeuo pipefail

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --file=/opt/zora/schema.sql

