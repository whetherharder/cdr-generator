FROM node:20-bookworm-slim

# System deps + GitHub CLI
RUN apt-get update && \
    apt-get install -y curl ca-certificates && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
      https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && apt-get install -y \
      git sudo ripgrep jq tmux gh openssh-client \
      python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

# node image has uid 1000 taken by `node` user — replace with `claude`
# uid 1000 matches typical host user so bind mounts work without permission issues
RUN userdel -r node && \
    useradd -m -s /bin/bash -u 1000 claude && \
    echo "claude ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER claude
WORKDIR /workspace

# Pre-create dirs so named volumes inherit correct ownership (claude:claude)
RUN mkdir -p /home/claude/.claude /home/claude/.config/gh /home/claude/.ssh /workspace/.venv && \
    chmod 700 /home/claude/.ssh

RUN echo '[ -f .venv/bin/activate ] && source .venv/bin/activate' >> /home/claude/.bashrc

ENTRYPOINT ["entrypoint.sh"]
CMD ["--dangerously-skip-permissions"]
