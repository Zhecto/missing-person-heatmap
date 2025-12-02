# syntax=docker/dockerfile:1
# Placeholder backend image definition. Update once the backend stack is selected.

# This backend is mainly for inference services.
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies here when the backend stack is finalized.
# Example:
# RUN apt-get update && apt-get install -y --no-install-recommends \ 
#     build-essential && rm -rf /var/lib/apt/lists/*

COPY ../../pyproject.toml ../../README.md ./

# TODO: install dependencies specific to the backend service.
# RUN pip install --no-cache-dir poetry && poetry install --no-root --only main

COPY ../../src ./src

CMD ["python", "-m", "http.server", "8080"]  # Replace with the actual backend entry point
