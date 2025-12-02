# syntax=docker/dockerfile:1
# Placeholder frontend image definition. Adjust to match the chosen UI stack.

# This frontend is mainly for user interaction.

FROM node:22-alpine AS base

WORKDIR /app

# Install dependencies once package.json exists.
# COPY ../../src/frontend/package.json ./
# RUN npm install --legacy-peer-deps

COPY ../../src/frontend ./src/frontend

CMD ["npm", "run", "dev"]  # Replace with the actual frontend start command
