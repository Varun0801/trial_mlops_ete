# Use Python 3.11 as the base image
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .
COPY models/ /app/models/


# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "fastapi_deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# docker build -f Docker/Dockerfile -t my-fastapi-app .
# docker run -d -p 8000:8000 my-fastapi-app
# netstat -ano --> fot listing the port numbers
# taskkill /PID 19160 /F --> for killing the process, give the PID number of the port number
# docker run -it --entrypoint /bin/bash my-fastapi-app