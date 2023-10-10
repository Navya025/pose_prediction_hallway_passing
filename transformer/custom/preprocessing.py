# Description: This file contains functions to process the raw data from the Kinect
# into a format that can be used by the model. The data is processed from a raw text
# file and written to a new text file with the processed data. The processed data is then
# loaded into a PyTorch Dataset object, which will then be loaded into a DataLoader object
# in order to train/validate the model.

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset, DataLoader

# Processes the entire file of raw data and writes it to a new file.
def process_data(filename):
    #read from the filename file in /data directory
    with open(filename, 'r') as f:
        data = f.read()
    
    #split the data into lines
    lines = data.split('\n')
    
    #process each line of data, add it to a string
    for line in lines:
            processed_frame = process_frame(line)
            final_string = ""
            #write the processed data to the new file in the specified format
            for i in range(0, len(processed_frame), 7):
                final_string+= "{processed_frame[i]}, {processed_frame[i+1]}, {processed_frame[i+2]}, {processed_frame[i+3]}, {processed_frame[i+4]}, {processed_frame[i+5]}, {processed_frame[i+6]}\n"
    
    
    #create a new file to write the processed data string to
    with open('processed_data.txt', 'w') as f:
        f.write("PROCESSED DATA: FORMAT = X, Y, Z, Angle, Axis X, Axis Y, Axis Z\n\n")
        f.write(final_string)
        f.write("\n\nEND OF PROCESSED DATA\n\n")
        f.close()

# Processes a single line of text from the raw data file, applying 
# Rodrigues rotation to the joint orientations

# param - frame is a single line of text from the data file 
# representing all the joints of a single frame
def process_frame(frame, num_joints=25, num_features=7):
    #split the frame into a list of numbers
    frame = frame.split()
    
    #convert each number from a string to a float
    frame = [float(num) for num in frame]
    
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
        
        #get the joint orientation (last four elements of each row)
        joint_orientation = frame[i, 3:]
        
        rodrigues_rotation = quat_to_rodrigues(joint_orientation)
        
        #positional components of final data for each joint in each frame
        processed_frame[i] = joint_position[0] #x
        processed_frame[i + 1] = joint_position[1] #y
        processed_frame[i + 2] = joint_position[2] #z
        #Rodrigues rotation formula
        processed_frame[i + 3] = rodrigues_rotation[0] #angle
        processed_frame[i + 4] = rodrigues_rotation[1][0] #axis x
        processed_frame[i + 5] = rodrigues_rotation[1][1] #axis y
        processed_frame[i + 6] = rodrigues_rotation[1][2] #axis z
    
    #return the processed frame
    return processed_frame

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

# Wrapper class for the Kinect dataset (operates on processed_data.txt)
class KinectDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()[2:-2]  # skipping the header and end notes
            self.data = [list(map(float, line.strip().split(','))) for line in lines]

    def __len__(self):
        return len(self.data) - 2  # adjust to avoid indexing out of range

    #gets a specific frame from the input data
    def __getitem__(self, index):
        sample_input = torch.tensor(self.data[index])  # get frame idx
        sample_target = torch.tensor(self.data[index + 2])  # get frame idx+2
        return sample_input, sample_target
