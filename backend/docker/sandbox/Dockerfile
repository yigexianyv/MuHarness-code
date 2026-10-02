FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        make \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/muharness-requirements/
RUN python -m pip install --no-cache-dir \
        -r /tmp/muharness-requirements/requirements-dev.txt \
    && rm -rf /tmp/muharness-requirements

RUN mkdir /workspace && chown 65532:65532 /workspace
USER 65532:65532
WORKDIR /workspace

CMD ["/bin/sh"]
