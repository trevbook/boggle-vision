# Stage 1: Build stage
# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Set the working directory in the builder stage
WORKDIR /usr/src/app

# Install PyTorch
RUN pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Stage 2: Runtime stage
# Use a slim version of Python for the runtime environment
FROM python:3.10-slim

# Set the working directory in the runtime container
WORKDIR /usr/src/app

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/src/app /usr/src/app

# Install uvicorn in the runtime container
RUN pip install uvicorn

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]