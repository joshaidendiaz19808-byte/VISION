FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY checks ./checks
COPY src ./src
COPY api ./api
COPY web ./web
RUN useradd -m vision && chown -R vision /app
USER vision
EXPOSE 8000
# Serves the API at /api and the web app at /. Set VISION_ALLOW_SERVER_AI_KEY=1 + ANTHROPIC_API_KEY
# only if you want to pay for users' AI reviews.
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
