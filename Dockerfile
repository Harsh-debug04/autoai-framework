# --- Build Stage ---
# Use a full Python image to build dependencies, as some might need build tools.
FROM python:3.10 as builder

# Set the working directory
WORKDIR /app

# Install poetry for dependency management
# Using Poetry is a robust way to handle dependencies, but for now, we'll stick to pip.
# The venv is created in a way that can be copied to the next stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Final Stage ---
# Use a slim image for the final application to reduce size.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY autoai/ ./autoai
COPY main.py .
COPY data/ ./data

# Expose the port the API will run on
EXPOSE 8000

# Define the command to run the application
# The host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "autoai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
