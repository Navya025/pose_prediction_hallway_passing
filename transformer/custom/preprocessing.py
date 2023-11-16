# Description: This file contains functions to process the raw data from the Kinect
# into a format that can be used by the model. The data is processed from a raw text
# file and written to a new text file with the processed data. The processed data is then
# loaded into a PyTorch Dataset object, which will then be loaded into a DataLoader object
# in order to train/validate the model.

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset, DataLoader

num_frames = 5 #number of frames in input/output sequences
GLOBAL_MIN = float('inf') #find min/max for normalization
GLOBAL_MAX = float('-inf')

# Processes the entire file of raw data and writes it to a new file.
def process_data(filename):
    global GLOBAL_MIN, GLOBAL_MAX
    #read from the filename file in /data directory
    with open(filename, 'r') as f:
        data = f.read()
    
    #split the data into lines
    lines = data.split('\n')
    final_string = ""
    #initial pass to find min/max for normalization
    for line in lines:
        frame = [float(num.strip()) for num in line[1:-1].split(',')]
        current_min = min(frame[::7])  # take every 7th value starting from 0th, which are x coordinates
        current_max = max(frame[::7])  
        GLOBAL_MIN = min(GLOBAL_MIN, current_min)
        GLOBAL_MAX = max(GLOBAL_MAX, current_max)

        current_min = min(frame[1::7])  # y coordinates
        current_max = max(frame[1::7])  
        GLOBAL_MIN = min(GLOBAL_MIN, current_min)
        GLOBAL_MAX = max(GLOBAL_MAX, current_max)

        current_min = min(frame[2::7])  # z coordinates
        current_max = max(frame[2::7])  
        GLOBAL_MIN = min(GLOBAL_MIN, current_min)
        GLOBAL_MAX = max(GLOBAL_MAX, current_max)
    
    #process each line of data, add it to a string
    for line in lines:
            processed_frame = process_frame(line)
            
            #write the processed data to the new file in the specified format
            final_string+= f"{processed_frame}\n"
    
    
    #create a new file to write the processed data string to
    with open('processed_train1.txt', 'w') as f:
        f.write("PROCESSED TRAINING DATA: FORMAT = X, Y, Z, Angle, Axis X, Axis Y, Axis Z\n\n")
        f.write(final_string)
        f.write("\n\nEND OF PROCESSED DATA\n\n")
        f.close()

# Processes a single line of text from the raw data file, applying 
# Rodrigues rotation to the joint orientations

# param - frame is a single line of text from the data file 
# representing all the joints of a single frame
def process_frame(frame, num_joints=32, num_features=7):
    global GLOBAL_MIN, GLOBAL_MAX
    #remove the first and last character in each line (brackets)
    frame = frame[1:-1]
    
    #split the frame into a list of numbers
    frame = [num.strip() for num in frame.split(',')]
    
    #convert each number from a string to a float
    frame = [float(num.strip()) for num in frame]
    
    #convert the list into a numpy array
    frame = np.array(frame)
    
    #reshape the array into a 2D array
    frame = frame.reshape(num_joints, num_features)
    
    #create a new 1-D array to hold the processed data
    processed_frame = np.zeros((num_joints * num_features))
    
    #loop through each joint
    for i in range(num_joints):
        #get the joint position (first three elements of each row)
        joint_position = frame[i, :3]
        joint_position = normalize_xyz(joint_position)
        #get the joint orientation (last four elements of each row)
        joint_orientation = frame[i, 3:]
        rodrigues_rotation = quat_to_rodrigues(joint_orientation)
        
        index = i * num_features

        #positional components of final data for each joint in each frame
        processed_frame[index] = joint_position[0] #x
        processed_frame[index + 1] = joint_position[1] #y
        processed_frame[index + 2] = joint_position[2] #z
        #Rodrigues rotation formula
        processed_frame[index + 3] = rodrigues_rotation[0] #angle
        processed_frame[index + 4] = rodrigues_rotation[1][0] #axis x
        processed_frame[index + 5] = rodrigues_rotation[1][1] #axis y
        processed_frame[index + 6] = rodrigues_rotation[1][2] #axis z
    
    #return the processed frame
    return processed_frame.tolist()
final_string = ""
# Converts Kinect rotation quaternion to Rodrigues rotation vector
def quat_to_rodrigues(quaternion):
    # Convert quaternion to Rotation object
    rot = Rotation.from_quat(quaternion)

    # Convert to rotation vector (Rodriguez rotation representation)
    rot_vec = rot.as_rotvec()

    # The direction (unit vector) of rot_vec gives the axis, and its magnitude gives the angle.
    axis = rot_vec / np.linalg.norm(rot_vec)
    angle = np.linalg.norm(rot_vec) 
    
    result = (angle, axis)
    return result

def normalize_xyz(data):
    # Check if data has 3 elements
    if len(data) != 3:
        raise ValueError("Expected an array with 3 elements.")
    
    # Compute min and max values
    min_val = min(data)
    max_val = max(data)
    
    # Apply min-max normalization
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data



# Wrapper class for the Kinect dataset (operates on processed_data.txt)
class KinectDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()[4:-2]  # skipping the header and end notes
            self.data = [list(map(float, line.strip()[1:-1].split(','))) for line in lines]

    def __len__(self):
        # Minus 9 because we're taking 5 frames for input and 5 for target.
        # Therefore, the last possible starting index is len(self.data) - 10
        return len(self.data) - (num_frames * 2 - 1)

    def __getitem__(self, index):
        # Input consists of frames from index to index + 4
        sample = self.data[index:index + 5]

        # Target consists of frames from index + 5 to index + 9
        target = self.data[index + num_frames:index + num_frames * 2]

        # Convert the sample and target to PyTorch tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return sample, target

    
if __name__ == "__main__":
    # Filename of raw data to process
    raw_data_file = "data/raw_train.txt"

    # Process the data
    process_data(raw_data_file)

    print(f"Data processing complete. Processed data saved to 'processed_train.txt'.")
