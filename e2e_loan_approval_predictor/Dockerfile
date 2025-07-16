# Use official Python 3.12 base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything from local 'app/' into container '/app/'
COPY ./app /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
