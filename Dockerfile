FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y \
    git curl sudo ca-certificates ripgrep jq tmux \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code
RUN useradd -m -s /bin/bash claude && \
    echo "claude ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER claude
WORKDIR /workspace
COPY --chown=claude:claude . .
RUN python3.11 -m venv .venv && \
    .venv/bin/pip install --no-cache-dir -e ".[dev]"
