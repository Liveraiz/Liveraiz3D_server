FROM python:3.11-slim

WORKDIR /app

# System deps for scientific/visual libs (matplotlib, trimesh, SimpleITK, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Prepare upload directory (bind mount to persist if needed)
RUN mkdir -p uploads/dicom

EXPOSE 5051

CMD ["python", "app.py"]
