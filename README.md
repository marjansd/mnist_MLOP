## Project overview:

This project demonstrates an end-to-end machine learning pipeline using the MNIST dataset, where a Logistic Regression model is built with PyTorch, served with FastAPI, and containerized with Docker. The training process is logged with Weights and Biases (wandb) for monitoring and analysis.

## Requirements 
- Python 3.8+
- Docker
- Wandb account

## Instruction for running the code locally
**Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/repositoryname.git
    cd repositoryname
    ```

2. **Set Up a Virtual Environment**:
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate  # On Windows, use myenv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI Application**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

5. **Test the API Endpoint**: Use curl or Postman to test the `/predict` endpoint by uploading an image file.

## Instruction for building and running the Docker container:

1. **Build the Docker Image**:
    ```bash
    docker build -t fastapi-mnist-model .
    ```

2. **Run the Docker Container**:
    ```bash
    docker run -p 8000:8000 fastapi-mnist-model
    ```


## URL with wandb report: 
View project at: https://wandb.ai/marjansd-iowa-state-university/mnist-mlops
View run at: https://wandb.ai/marjansd-iowa-state-university/mnist-mlops/runs/6g01sxrw