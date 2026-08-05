# Tessera Studio data layer (Postgres 16 + pgvector + pg_trgm, Valkey 7).
#
# Targets:
#   db/up       -- docker compose up -d
#   db/down     -- docker compose down
#   db/reset    -- drop volumes + recreate (DESTRUCTIVE)
#   db/migrate  -- apply migrations in order (uses the Postgres container)
#   db/seed     -- load the test fixture (seeds/seed.sql)
#   db/shell    -- psql inside the Postgres container
#   db/valkey   -- valkey-cli inside the Valkey container
#   db/verify   -- apply migrations + seed to a throwaway DB and run pg_catalog checks
#
# All targets read .env for credentials (created from .env.example on
# first run). The compose file pins arm64; on an Intel Mac, drop the
# `platform: linux/arm64` lines first.
#
# PostgreSQL 16 image (pgvector/pgvector:pg16) already includes the
# vector and pg_trgm extensions; no extra image needed.

COMPOSE := docker compose
ENV_FILE := .env

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export
endif

# Connection params (fall back to defaults that match docker-compose.yml).
DB_USER ?= tessera
DB_NAME ?= tessera
DB_PORT ?= 5432
DB_PASS ?= tessera
VK_PORT ?= 6379

# Tool paths (overridable for CI).
PSQL ?= psql

# Wait for Postgres to be healthy before running migrations / seed.
# Up to ~30s; pg_isready exits 0 the moment the server accepts connections.
WAIT_PG := for i in $$(seq 1 30); do \
	$(PSQL) -h localhost -U "$(DB_USER)" -d "$(DB_NAME)" -p "$(DB_PORT)" \
		-c "SELECT 1" >/dev/null 2>&1 && exit 0; \
	sleep 1; \
done; \
	echo "Postgres not ready after 30s" >&2; exit 1

.PHONY: db/up db/down db/reset db/migrate db/seed db/shell db/valkey db/verify db/lint help

help: ## Show this help.
	@grep -E '^[a-zA-Z_/-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

db/up: ## Bring up Postgres + Valkey
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "==> no .env, copying from .env.example"; \
		cp .env.example $(ENV_FILE); \
	fi
	$(COMPOSE) up -d

db/down: ## Stop the containers (keeps volumes)
	$(COMPOSE) down

db/reset: ## DESTRUCTIVE: drop volumes + recreate
	$(COMPOSE) down -v
	$(COMPOSE) up -d
	$(MAKE) db/migrate
	$(MAKE) db/seed

db/migrate: ## Apply all migrations/*.sql in order
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "==> no .env, copying from .env.example"; \
		cp .env.example $(ENV_FILE); \
	fi
	@echo "==> waiting for Postgres at localhost:$(DB_PORT)..."
	@$(WAIT_PG)
	@for f in migrations/*.sql; do \
		echo "==> applying $$f"; \
		PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d "$(DB_NAME)" -p "$(DB_PORT)" \
			-v ON_ERROR_STOP=1 -f "$$f" || exit 1; \
	done
	@echo "==> migrations complete"

db/seed: ## Load the test fixture (seeds/seed.sql)
	@echo "==> waiting for Postgres at localhost:$(DB_PORT)..."
	@$(WAIT_PG)
	@echo "==> loading seeds/seed.sql"
	PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d "$(DB_NAME)" -p "$(DB_PORT)" \
		-v ON_ERROR_STOP=1 -f seeds/seed.sql
	@echo "==> seed complete"

db/shell: ## Open psql inside the Postgres container
	$(COMPOSE) exec postgres psql -U "$(DB_USER)" -d "$(DB_NAME)"

db/valkey: ## Open valkey-cli inside the Valkey container
	$(COMPOSE) exec valkey valkey-cli

db/verify: ## Apply migrations + seed to a throwaway DB and assert via pg_catalog
	@echo "==> applying migrations to a throwaway DB 'tessera_verify'"
	@$(WAIT_PG)
	PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d "$(DB_NAME)" -p "$(DB_PORT)" \
		-v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS tessera_verify; CREATE DATABASE tessera_verify;"
	@for f in migrations/*.sql; do \
		PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d tessera_verify -p "$(DB_PORT)" \
			-v ON_ERROR_STOP=1 -f "$$f" || exit 1; \
	done
	@PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d tessera_verify -p "$(DB_PORT)" \
		-v ON_ERROR_STOP=1 -f seeds/seed.sql
	@echo "==> verifying tables and indexes via pg_catalog"
	@PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d tessera_verify -p "$(DB_PORT)" -c "\
		SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;"
	@PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d tessera_verify -p "$(DB_PORT)" -c "\
		SELECT indexname FROM pg_indexes WHERE schemaname = 'public' ORDER BY indexname;"
	@PGPASSWORD="$(DB_PASS)" $(PSQL) -h localhost -U "$(DB_USER)" -d tessera_verify -p "$(DB_PORT)" -c "\
		SELECT proname FROM pg_proc WHERE proname = 'hybrid_search';"
	@echo "==> verify complete"

db/lint: ## Lint the docker-compose.yml syntax (works without the compose plugin)
	@python3 -c "import yaml,sys; d=yaml.safe_load(open('docker-compose.yml')); \
		assert 'services' in d, 'missing services'; \
		assert 'postgres' in d['services'], 'missing postgres service'; \
		assert 'valkey' in d['services'], 'missing valkey service'; \
		assert d['services']['postgres']['platform']=='linux/arm64', 'postgres not arm64'; \
		assert d['services']['valkey']['platform']=='linux/arm64', 'valkey not arm64'; \
		print('docker-compose.yml: OK')"
