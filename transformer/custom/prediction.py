from preprocessing import KinectDataset
from preprocessing import process_frame
from model import PosePredictionTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time
import os

BATCH_SIZE = 32

num_layers = 3
d_model = 512
num_heads = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PosePredictionTransformer(num_layers, d_model, num_heads).to(device)

# processed_data_file = "data/static_input.txt"
processed_data_file = "data/processed_val.txt"

input_dataset = KinectDataset(processed_data_file)
input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=True)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

file = open("output/output.txt", "w")

with torch.no_grad():
    for file in input_dataloader:
        for batch in file:
            for five_frames in batch:
                print("data")
                print(five_frames)
        # for data in batch:

        #     data = data.to(device)
        #     outputs = model(data)

        #     file.write("Given five frames\n")
        #     for frame in outputs:
        #         file.write(f"{frame}\n\n")

print("\nDone\n")
