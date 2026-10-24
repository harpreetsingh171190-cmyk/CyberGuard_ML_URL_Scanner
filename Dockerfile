FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-extra.txt ./
RUN pip install --no-cache-dir -r requirements.txt streamlit fastapi uvicorn
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
