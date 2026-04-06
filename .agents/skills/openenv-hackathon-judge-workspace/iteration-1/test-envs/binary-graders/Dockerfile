FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install pydantic fastapi uvicorn
CMD ["python", "-c", "print('Running')"]
