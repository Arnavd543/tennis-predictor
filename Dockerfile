FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps + Node.js for Tailwind build
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 postgresql-client curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

# Node dependencies (Tailwind build only)
COPY package.json ./
RUN npm install --save-dev

# Copy project
COPY . .

# Build Tailwind CSS (minified for production)
RUN npx tailwindcss \
    -i predictor/static/predictor/src/input.css \
    -o predictor/static/predictor/dist/styles.css \
    --minify

# Create directories
RUN mkdir -p /app/data /app/predictor/models /app/staticfiles /app/logs

# Collect static files
RUN python manage.py collectstatic --noinput

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/', timeout=5)"

# wait_for_db → migrate → gunicorn
CMD ["sh", "-c", "python manage.py wait_for_db && python manage.py migrate --noinput && gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 tennis_predictor.wsgi:application"]
