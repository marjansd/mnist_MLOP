#Dockerfile

# 1. Choose a base image with Python installed
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the necessary files into the container
# Copy the FastAPI application file
COPY app.py /app/
# Copy the trained model file (make sure model.pth is in the same directory as the Dockerfile)
COPY model.pth /app/
COPY model.py /app/ 
# Copy the requirements file for dependencies
COPY requirements.txt /app/

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
ENV WANDB_API_KEY 17258e4b0708335e1f31211b1c4f8ef55021aa06
# 5. Expose the port that FastAPI will run on
EXPOSE 8000

# 6. Set the command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# RUN
# docker run -p 8000:8000 fastapi-mnist-model