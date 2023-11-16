# Description: Train the model on the training data and validate on the validation data.

from secondpreprocessing import KinectDataset
from secondmodel import PosePredictionTransformer
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 0
LEARNING_RATE = 0.001
num_frames = 5

# files to read from
train_file = "../data/processed_train2.txt"
val_file = "../data/processed_val2.txt"
test_file = "../data/midterm-processed/curve-right_processed.txt"


def getGlobalMaxima(file):
    with open(file, "r") as f:
        lines = f.readlines()[1:]  # Skip first line of the file
    maxima = []
    for i in range(
        2
    ):  # Corrected syntax here (use 'for i in range(2):' instead of '(i in range(2))')
        line = lines[i]
        discarded, num = line.split(":")  # Discard the part before the colon
        maxima.append(float(num.strip()))
    return maxima[0], maxima[1]  # maxima[0] = globalMin, maxima[1] = globalMax

# denormalize the min-max normalization of the data
# array is an array of length 3 of x,y,z coordinates
def denormalize(arr, globalMin, globalMax):
    result = []
    for elem in arr:
        elem = elem * (globalMax - globalMin) + globalMin
        elem /= 10  # convert from cm to mm
        result.append(elem)
    return result

def calculate_average_distance(outputs, target, globalMin, globalMax):
    total_distance = 0.0
    total_count = 0
    for i in range(outputs.size(2)):
        output = outputs[:, :, i]
        tgt = target[:, :, i]  # these are num_framesx3 matrices
        for j in range(num_frames):
            outputFrame = output[j, :]
            tgtFrame = tgt[j, :]
            outputFrame = denormalize(outputFrame, globalMin, globalMax)
            tgtFrame = denormalize(tgtFrame, globalMin, globalMax)

            dist = math.sqrt(  # apply 3D distance formula
                (outputFrame[0] - tgtFrame[0]) ** 2
                + (outputFrame[1] - tgtFrame[1]) ** 2
                + (outputFrame[2] - tgtFrame[2]) ** 2
            )
            total_distance += dist  # the distance from the ground truth
            total_count += 1

    average_distance = total_distance / total_count if total_count else 0
    return average_distance

# global min/max
globalMin, globalMax = getGlobalMaxima(train_file)

# Adjust model parameters here
num_layers = 3
d_model = 32
num_heads = 8

# threshold for correct prediction (in cm)
threshold = 10

# TensorBoard writer
writer = SummaryWriter()

# Load training and validation data
train_dataset = KinectDataset(train_file)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = KinectDataset(val_file)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = KinectDataset(test_file)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# adjust parameters here
model = PosePredictionTransformer(num_layers, d_model, num_heads).to(device)
# most commonly used regression problem loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, "min", factor=0.1, patience=10, verbose=True
# )

lowest_avg_distance = float("inf")  # To keep track of the best validation accuracy

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
    total_avg_distance = 0.0
    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            # Calculate average distance
            avg_distance = calculate_average_distance(
                outputs, target, globalMin, globalMax
            )
            total_avg_distance += avg_distance

        val_loss_epoch = val_loss / len(val_dataloader)
        total_avg_distance /= len(val_dataloader)  # Average over all batches

        # Log validation loss and average distance to TensorBoard
        writer.add_scalar("Loss/validation", val_loss_epoch, epoch)
        writer.add_scalar("Average Distance/validation", total_avg_distance, epoch)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Validation Loss: {val_loss_epoch}, Average Distance: {total_avg_distance} cm"
        )

    # scheduler.step(val_loss_epoch)

    # Save the best model
    if total_avg_distance < lowest_avg_distance:
        lowest_avg_distance = total_avg_distance
        torch.save(model.state_dict(), "lowest_avg.pth")
        print("Best model saved with MAE:", lowest_avg_distance)

print("Training finished!")

# Testing loop
model.load_state_dict(torch.load("lowest_avg.pth"))
model.eval()
test_loss = 0.0
total_avg_distance = 0.0
with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()

        # Calculate average distance
        avg_distance = calculate_average_distance(outputs, target, globalMin, globalMax)
        total_avg_distance += avg_distance

    test_loss_epoch = test_loss / len(test_dataloader)
    total_avg_distance /= len(test_dataloader)  # Average over all batches

    # Log testing loss and average distance to TensorBoard
    writer.add_scalar("Loss/test", test_loss_epoch, EPOCHS)
    writer.add_scalar("Average Distance/test", total_avg_distance, EPOCHS)

    print(f"Testing Loss: {test_loss_epoch}, Average Distance: {total_avg_distance} cm")

print("Testing finished!")
