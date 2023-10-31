# Description: Train the model on the training data and validate on the validation data.

from secondpreprocessing import KinectDataset
from secondmodel import PosePredictionTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 0.001

# Adjust model parameters here
num_layers = 3
d_model = 32
num_heads = 8

# TensorBoard writer
writer = SummaryWriter()

# Load training and validation data
train_dataset = KinectDataset("../data/processed_train2.txt")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = KinectDataset("../data/processed_val2.txt")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# adjust parameters here
model = PosePredictionTransformer(num_layers, d_model, num_heads).to(device)
# most commonly used regression problem loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, "min", factor=0.1, patience=10, verbose=True
#)


best_accuracy = 0.0  # To keep track of the best validation accuracy

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Training loop
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1
        )  # Gradient clipping
        optimizer.step()

        running_loss += loss.item()

        # Log training loss to TensorBoard
        writer.add_scalar("Loss/train", running_loss / len(train_dataloader), epoch)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {running_loss / len(train_dataloader)}"
    )

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    accuracy = 0.0
    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            # Calculate validation accuracy
            # within 1% of target value = success
            correct += torch.sum(torch.abs(outputs - target) < 0.01)
            total += target.numel()

        val_loss_epoch = val_loss / len(val_dataloader)
        accuracy = 100 * correct / total

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar("Loss/validation", val_loss_epoch, epoch)
        writer.add_scalar("Accuracy/validation", accuracy, epoch)

    print(f"Validation Loss: {val_loss_epoch}, Validation Accuracy: {accuracy}%")
    #scheduler.step(val_loss_epoch)

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved with accuracy:", best_accuracy)

print("Training finished!")


# Load testing data
test_dataset = KinectDataset("../data/test.txt")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ... [training and validation loops]

print("Training finished!")
print("Testing model...")
# Testing loop
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0.0
correct = 0
total = 0
accuracy = 0.0
with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()

        # Calculate testing accuracy: within 1% of target value = success
        correct += torch.sum(torch.abs(outputs - target) < 0.05)
        total += target.numel()

    test_loss_epoch = test_loss / len(test_dataloader)
    accuracy = 100 * correct / total

    # Log testing loss and accuracy to TensorBoard
    writer.add_scalar("Loss/test", test_loss_epoch, EPOCHS)  # Note: I'm using EPOCHS as the x-value
    writer.add_scalar("Accuracy/test", accuracy, EPOCHS)

print(f"Testing Loss: {test_loss_epoch}, Testing Accuracy: {accuracy}%")
