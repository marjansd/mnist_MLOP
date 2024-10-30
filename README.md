## Project overview:

This project demonstrates an end-to-end machine learning pipeline using the MNIST dataset, where a Logistic Regression model is built with PyTorch, served with FastAPI, and containerized with Docker. The training process is logged with Weights and Biases (wandb) for monitoring and analysis.

## Requirements 
- Python 3.8+
- Docker
- Wandb account

## Instruction for running the code locally
1.Clone the Repository:
    ```bash
    git clone https://github.com/marjansd/mnist_MLOP.git
    cd repositoryname
    ```

2. Set Up a Virtual Environment:
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate  # On Windows, use myenv\Scripts\activate
    ```

3. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run model.py    :python3 model.py
5. Run app.py      :python3 app.py
6. Instruction for building and running the Docker container:

6.1. Build the Docker Image:
    ```bash
    docker build -t fastapi-mnist-model .
    ```

6.2. Run the Docker Container:
    ```bash
    docker run -p 8000:8000 fastapi-mnist-model
    ```
## URL with wandb report: 
View project at: https://wandb.ai/marjansd-iowa-state-university/mnist-mlops
View run at: https://wandb.ai/marjansd-iowa-state-university/mnist-mlops/runs/6g01sxrw
