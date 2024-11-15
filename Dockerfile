# Use an official Python runtime as a parent image
FROM --platform=amd64 python:3.10

# Set environment variables for Poetry
ENV POETRY_VERSION=1.4.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install poetry
RUN poetry lock
RUN poetry install
RUN poetry run task setup
# RUN pip install vllm==0.6.3

# Expose the port that Gradio will run on
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the command to run the Gradio app
CMD ["python", "app.py"]
