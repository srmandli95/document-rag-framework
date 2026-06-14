.PHONY: install backend ui build run stop logs test health clean

install:
	pipenv install --dev

backend:
	cd backend && DEV_AUTH_DISABLED=$${DEV_AUTH_DISABLED:-true} DEV_AUTH_USER_ID=$${DEV_AUTH_USER_ID:-local-user-123} PYTHONPATH=. pipenv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ui:
	cd frontend && npm run dev

build:
	docker compose up -d --build

run:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

health:
	curl http://localhost:8000/health

test:
	docker compose exec backend sh -c "PYTHONPATH=/app pytest -q"
