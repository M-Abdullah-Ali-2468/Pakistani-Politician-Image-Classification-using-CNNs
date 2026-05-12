# Deployment Container

This folder is a standalone model serving container for the Pakistani Politician classifier.

## Structure

```
deployment/
|-- app/
|   |-- main.py
|   |-- inference.py
|   `-- utils.py
|-- model/
|   `-- trained_model.pth
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Run Locally

```bash
cd deployment
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

## Build Docker Image

```bash
docker build -t pakistani-politician-classifier:latest ./deployment
docker run -p 7860:7860 pakistani-politician-classifier:latest
```

## Hugging Face Spaces

Use Docker SDK for the Space and set the build context to this `deployment/` directory.
