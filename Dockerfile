FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py trades_sample.csv ./
COPY checks ./checks
COPY src ./src
COPY .streamlit/config.toml ./.streamlit/config.toml
RUN useradd -m vision && chown -R vision /app
USER vision
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
