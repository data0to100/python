PYTHON=python
PIP=pip
APP=app

.PHONY: install run test format lint docker-build docker-run

install:
	$(PIP) install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

docker-build:
	docker build -t data-analysis-pro-aapc .

docker-run:
	docker run -it --rm -p 8000:8000 --env-file .env data-analysis-pro-aapc