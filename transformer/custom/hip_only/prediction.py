import torch
import numpy as np
from torch.utils.data import DataLoader
from secondmodel import PosePredictionTransformer
from secondpreprocessing import KinectDataset

# Function to make predictions with the model
def make_predictions(model, dataloader, device):
    predictions = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (input_seq, _) in enumerate(dataloader):
            input_seq = input_seq.to(device)
            output = model(input_seq)  # Get model predictions
            predictions.append(output.cpu().numpy())  # Store predictions
    return predictions

# Combine predictions with overlapping averaging
def average_overlapping_predictions(predictions, overlap_size):
    averaged_predictions = []
    for i in range(len(predictions)):
        for j in range(5):  # Assuming each prediction has 5 frames
            if i == 0 and j < overlap_size:
                # Initialize the first set of predictions
                averaged_predictions.append(predictions[i][j])
            elif j < overlap_size:
                # Average the overlapping predictions
                index = i + j
                averaged_predictions[index] = (averaged_predictions[index] + predictions[i][j]) / 2
            else:
                # Add the non-overlapping predictions
                averaged_predictions.append(predictions[i][j])
    return averaged_predictions

# Placeholder paths
model_path = 'lowest_avg.pth'
data_path = '../data/midterm-processed/curve-left_processed.txt'
output_path = '../../../visualization/joint_visualizations/left-predict.txt'

# Model parameters (adjust as necessary)
num_layers = 3
d_model = 32
num_heads = 8

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PosePredictionTransformer(num_layers, d_model, num_heads)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Load the data using KinectDataset
dataset = KinectDataset(data_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Make predictions
predictions = make_predictions(model, dataloader, device)
predictions = np.array(predictions)
predictions = predictions.reshape(predictions.shape[0], 5, 3)

# Average the overlapping predictions
overlap_size = 4  # Number of frames that overlap between predictions
averaged_predictions = average_overlapping_predictions(predictions, overlap_size)

# Save predictions to file
with open(output_path, 'w') as f:
    for pred in averaged_predictions:
        f.write("[")
        # Assuming each prediction is a numpy array with a shape of (num_features,)
        pred_string = ','.join(map(str, pred))
        f.write(pred_string + ']' + '\n')
print(f"Predictions saved to {output_path}")
