FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install pydantic fastapi uvicorn
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
