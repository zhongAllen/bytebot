# syntax=docker/dockerfile:1

# Base image
FROM node:20-alpine

# Create app directory
WORKDIR /app

# Copy monorepo subpackages needed by agent
# Context is packages/, so these paths exist:
#  - ./shared
#  - ./bytebot-agent
COPY ./shared ./shared
COPY ./bytebot-agent/ ./bytebot-agent/

# Work inside agent
WORKDIR /app/bytebot-agent

# Install dependencies
# Prefer npm ci if lockfile present, fallback to npm install
RUN [ -f package-lock.json ] && npm ci || npm install

# Build (will also build ../shared via agent scripts where needed)
RUN npm run build

# Expose default port (NestJS default 3000 unless overridden)
EXPOSE 3000

# Start in production (script already handles prisma migrate deploy/generate)
CMD ["npm", "run", "start:prod"]