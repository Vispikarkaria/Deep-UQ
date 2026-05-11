FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README_PYPI.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[tests,benchmarks]"

COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY data/ data/

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
