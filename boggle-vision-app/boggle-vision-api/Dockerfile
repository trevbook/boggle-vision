# Start from the base image of Python
FROM python:3.11

# Set the working directory to /code
WORKDIR /code

# Copy the requirements.txt file to /code
COPY ./requirements.txt /code/requirements.txt

# Install the requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory to /code/app
COPY ./app /code/app

# Run the command to start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
