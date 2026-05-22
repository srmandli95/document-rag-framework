.PHONY: install backend ui build run stop logs test health clean

install:
	pipenv install --dev

backend:
	cd backend && PYTHONPATH=. pipenv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ui:
	cd chainlit_ui && BACKEND_BASE_URL=http://localhost:8000 pipenv run chainlit run app.py --host 0.0.0.0 --port 8501

build:
	docker compose up --build

run:
	docker compose up

stop:
	docker compose down

logs:
	docker compose logs -f

test:
	cd backend && PYTHONPATH=. pipenv run pytest -q

health:
	curl http://localhost:8000/health

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
