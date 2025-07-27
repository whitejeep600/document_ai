FROM python:3.9

RUN mkdir /poetry /venv document_ai

RUN python -m venv /poetry /venv

WORKDIR /document_ai

COPY poetry.lock pyproject.toml ./

ARG poetry_version=1.5.1
RUN /poetry/bin/pip install --no-cache-dir poetry==${poetry_version} \
    && /poetry/bin/poetry --version

RUN . /venv/bin/activate \
    && /poetry/bin/poetry lock --check \
    && /poetry/bin/poetry install --no-root

ENV PATH="/venv/bin:$PATH"

COPY src src

EXPOSE 8000

CMD ["uvicorn", "src.app_main:document_ai", "--host", "0.0.0.0", "--port", "8000"]