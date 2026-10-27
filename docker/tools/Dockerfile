# Sandbox for reading untrusted NTSB files: mdbtools (Access .mdb -> CSV) and
# poppler + tesseract (OCR fallback). Run with --network none and read-only inputs.
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends mdbtools poppler-utils tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*
ENV LC_ALL=C.UTF-8
RUN useradd --create-home tools
USER tools
WORKDIR /work
