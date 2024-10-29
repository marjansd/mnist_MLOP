
# Model deploymenet script

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
from model import LogisticRegression


app = FastAPI()

# 1. Instantiate the model:
model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=[128, 64], dropout_rate=0.2)

# 2. Load the state_dict into the model instance:
model.load_state_dict(torch.load("model.pth")) 

# 3. Set the model to evaluation mode:
model.eval()  

# Define the image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize the image to 28x28
    transforms.ToTensor(),        # Convert image to a tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST dataset
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file and convert it to a grayscale image
    image = Image.open(io.BytesIO(await file.read())).convert('L')

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    # Return the prediction
    return {"prediction": prediction}


# Running the API    
import uvicorn
import nest_asyncio

# Apply the patch for nested event loops
nest_asyncio.apply()

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)