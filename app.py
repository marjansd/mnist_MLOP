from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
from model import LogisticRegression

app = FastAPI()


# Load the trained model and set it to evaluation mode
model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=[128, 64], dropout_rate=0.2)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))  # Load the trained weights
model.eval()  # Set the model to evaluation mode

# Preprocessing pipeline for incoming images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    return {"prediction": prediction}

# Run this script with: uvicorn app:app --host 0.0.0.0 --port 8000

# Running the API    
import uvicorn
import nest_asyncio

# Apply the patch for nested event loops
nest_asyncio.apply()

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)

# To run  docker run -p 8000:8000 fastapi-mnist-mode    