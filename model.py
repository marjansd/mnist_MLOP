import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
from sklearn.metrics import precision_score, recall_score


# Model definition
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(LogisticRegression, self).__init__()
        layers = []

        # Input layer
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.network(x)

# Hyperparameters and configuration
config = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 64,
    "hidden_layers": [128, 64],
    "dropout_rate": 0.2
}

# Initialize wandb and watch the model for logging parameters and gradients
wandb.init(project="mnist-mlops", config=config)
#wandb.watch(model, log="all")

# Data loading and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize model, loss, and optimizer
input_dim = 28 * 28
output_dim = 10
model = LogisticRegression(input_dim, output_dim, config["hidden_layers"], config["dropout_rate"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
wandb.watch(model, log="all")

# Function to log sample predictions
def log_predictions(model, data, target, num_samples=10):
    model.eval()
    with torch.no_grad():
        output = model(data[:num_samples])
        predictions = output.argmax(dim=1, keepdim=True)
    # Log images with predictions vs actual labels
    wandb.log({"Predictions": [wandb.Image(data[i], caption=f"Pred: {predictions[i].item()}, Actual: {target[i].item()}")
                               for i in range(num_samples)]})

# Training loop with detailed logging
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Reset gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Accumulate loss
        total_loss += loss.item()

        # Log additional metrics like learning rate and gradient norms
        wandb.log({
            "Learning Rate": optimizer.param_groups[0]["lr"],
            "Gradient Norm": sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        })

    # Log average training loss per epoch
    avg_loss = total_loss / len(train_loader)
    wandb.log({"Training Loss": avg_loss, "epoch": epoch + 1})

    # Validation with detailed logging of precision and recall
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Calculate accuracy, precision, and recall
    accuracy = 100. * correct / len(test_loader.dataset)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    # Log accuracy, precision, and recall
    wandb.log({
        "Validation Accuracy": accuracy,
        "Validation Precision": precision,
        "Validation Recall": recall,
        "epoch": epoch + 1
    })
    print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Log sample predictions at the end of each epoch
    log_predictions(model, next(iter(test_loader))[0], next(iter(test_loader))[1])

wandb.finish()

import torch

# Define your model structure to match the original model
model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=[128, 64], dropout_rate=0.2)

# Load the weights (state_dict) into the model
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model to evaluation mode


