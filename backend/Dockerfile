FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

RUN addgroup --system factchecker && adduser --system --ingroup factchecker factchecker
ARG INSTALL_ML=false
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/app ./app
COPY backend/training ./training
COPY backend/evaluation ./evaluation
RUN pip install --upgrade pip \
    && if [ "$INSTALL_ML" = "true" ]; then pip install '.[ml]'; else pip install .; fi

COPY configs /app/configs
COPY datasets /app/datasets
RUN chown -R factchecker:factchecker /app
USER factchecker

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
