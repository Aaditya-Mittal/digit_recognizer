FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
