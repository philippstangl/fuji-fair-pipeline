FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code last because changes most often
COPY pipeline/ ./pipeline/
COPY config.yaml ./
# Ad-hoc tools
COPY scripts/ ./scripts/

RUN mkdir -p /app/data /app/logs /app/runs

ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app 2>/dev/null || true && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash app 2>/dev/null || true && \
    chown -R ${APP_UID}:${APP_GID} /app
USER ${APP_UID}:${APP_GID}

ENTRYPOINT ["python", "-m", "pipeline"]
CMD []
