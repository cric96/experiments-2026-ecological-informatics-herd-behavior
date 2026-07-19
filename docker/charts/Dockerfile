FROM python:3.12.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /experiment

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy configuration and lockfiles first to leverage docker layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install project dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source
COPY process.py ./

# Run the script using uv run
CMD ["uv", "run", "process.py"]
