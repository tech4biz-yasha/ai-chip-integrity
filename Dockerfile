# AI Chip Integrity Suite: every probe and its self-test in one container.
#
#   docker run --rm --gpus all -v "$PWD:/results" ghcr.io/tech4biz-yasha/ai-chip-integrity:0.4.0
#
# Base: the same PyTorch 2.8.0 and CUDA 12.8 build the published H100 rows were measured with.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.title="ai-chip-integrity" \
      org.opencontainers.image.description="Open, vendor-neutral tests for silent computation errors in AI chips" \
      org.opencontainers.image.source="https://github.com/tech4biz-yasha/ai-chip-integrity" \
      org.opencontainers.image.url="https://chipintegrity.org/" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /opt/ai-chip-integrity
COPY pyproject.toml README.md LICENSE ./
COPY chip_integrity ./chip_integrity
RUN pip install --no-cache-dir . && chip-integrity --version

WORKDIR /results
ENTRYPOINT ["chip-integrity"]
CMD ["run", "--out-dir", "/results"]
