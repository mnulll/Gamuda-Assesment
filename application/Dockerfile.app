# ---- Streamlit frontend --------------------------------------------------- #
FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# API_URL is injected by docker-compose so the frontend can find the backend.
ENV API_URL=http://api:8000
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
