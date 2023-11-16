from preprocessing import KinectDataset
from preprocessing import process_frame
from model import PosePredictionTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time
import os

num_layers = 3
d_model = 512
num_heads = 8
seq_len = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PosePredictionTransformer(num_layers, d_model, num_heads).to(device)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

raw_data_file = "data/stream.txt"
file = open("output/output.txt", "w")

last_frame = -1
                 
while True:
    if os.path.exists(raw_data_file):
        with open(raw_data_file, "r") as file:
            lines = file.readlines()
            if len(lines) < 5:
                time.sleep(1) # wait for at least five frames of data
                continue

            current_frames = len(lines)
            frame_data = []

            for line in lines:
                array = json.loads(line)
                processed_frame = process_frame(array)
                frame_data.append(processed_frame)
            
            if current_frames >= last_frame + 5:
                last_frame = current_frames - 1

                # use the last five frames to make a prediction
                frames_to_predict_from = frame_data[last_frame - 4:last_frame + 1] 
                frames_to_predict_from = torch.tensor(frames_to_predict_from, dtype=torch.float32)

                with torch.no_grad():
                    # predict five frames (maybe 100?)
                    prediction = model(frames_to_predict_from)

                    for frame in prediction:
                        file.write(f"{frame}\n\n")
           
            # end of stream
            if current_frames == 100:
                break

    time.sleep(1) # check the file every second

