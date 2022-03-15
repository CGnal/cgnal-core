# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster as builder

WORKDIR /app

COPY requirements/requirements.in requirements/requirements.in
COPY requirements/requirements_ci.txt requirements/requirements_ci.txt
COPY LICENSE MANIFEST.in versioneer.py setup.py setup.cfg README.md .
COPY cgnal cgnal
COPY tests tests
RUN pip install -r requirements/requirements_ci.txt
RUN python3 -m pytest
RUN python3 setup.py sdist

FROM python:3.7-slim-buster
WORKDIR /app
COPY --from=builder /app/dist /app/dist
RUN ls -t ./dist/*.tar.gz | xargs pip install
ENTRYPOINT ["python"]
