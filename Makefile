.PHONY: dev test lint backend frontend

dev:
	docker compose up --build

test:
	cd backend && pytest
	cd frontend && npm test -- --run

lint:
	cd backend && ruff check . && mypy app training evaluation
	cd frontend && npm run lint && npm run typecheck

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev
