# This file preprocesses the dataset outlined in https://www.sciencedirect.com/science/article/pii/S2352340923004523
# For use in training a transformer model which predicts pose based on previous joint pose data

# The dataset is organized as follows:
"""
Frame {
    
    Joint orientation {
        Joint 1 quaternion1
        Joint 1 quaternion2
        Joint 1 quaternion3
        Joint 1 quaternion4
        ...
        Joint 32 quaternion1
        Joint 32 quaternion2
        Joint 32 quaternion3
        Joint 32 quaternion4
    }
    
    Joint position {
        Joint 1 X
        Joint 1 Y
        Joint 1 Z
        ...
        Joint 32 X
        Joint 32 Y
        Joint 32 Z
    }
}
"""
import json
import os
import glob

def process_frames_to_txt(frames, output_file_path):
    output_lines = []
    for frame in frames:
        if frame["bodies"]:
            for body in frame["bodies"]:
                joint_positions = body["joint_positions"]
                joint_orientations = body["joint_orientations"]
                frame_data = [','.join(map(str, pos + ori)) for pos, ori in zip(joint_positions, joint_orientations)]
                output_lines.append(f"[{','.join(frame_data)}]")

    with open(output_file_path, "w") as txt_file:
        txt_file.write('\n'.join(output_lines))

def process_directory(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_path in glob.glob(os.path.join(input_folder_path, '**', '*.json'), recursive=True):
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        frames_data = json_data['frames']
        output_file_name = os.path.basename(file_path).replace('.json', '.txt')
        output_file_path = os.path.join(output_folder_path, output_file_name)
        process_frames_to_txt(frames_data, output_file_path)
        print(f"Processed {file_path} to {output_file_path}")

# Paths for the input and output directories
input_folder_path = './Posner_Dataset/SUBJECTS'
output_folder_path = './Posner_Dataset/processed'

# Process all JSON files in the directory
process_directory(input_folder_path, output_folder_path)



