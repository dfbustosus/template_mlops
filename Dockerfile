FROM python:3.11-slim

WORKDIR /app

# Install runtime deps
COPY pyproject.toml pyproject.toml
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

COPY src/ src/

CMD ["python", "-m", "mlops_template.cli"]
