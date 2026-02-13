FROM python:3.11-slim

WORKDIR /app

# Install only web-app dependencies (no torch/training deps)
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy application code
COPY app/ app/
COPY models/openai_backend.py models/openai_backend.py
COPY models/prompts.py models/prompts.py
COPY models/__init__.py models/__init__.py
COPY qa/mvp_utils.py qa/mvp_utils.py
COPY qa/__init__.py qa/__init__.py
COPY pyproject.toml .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
