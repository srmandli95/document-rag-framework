.PHONY: install backend ui build run stop logs test health clean

install:
	pipenv install --dev

backend:
	cd backend && PYTHONPATH=. pipenv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ui:
	cd chainlit_ui && BACKEND_BASE_URL=http://localhost:8000 pipenv run chainlit run app.py --host 0.0.0.0 --port 8501

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