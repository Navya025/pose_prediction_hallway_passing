# Description: Version of the preprocessing script that

import numpy as np
import os
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset, DataLoader

num_frames = 5  # number of frames in input/output sequences
GLOBAL_MIN = float("inf")  # find min/max for normalization
GLOBAL_MAX = float("-inf")
final_string = ""


# Processes the entire file of raw data and writes it to a new file.
def process_data(filename):
    global GLOBAL_MIN, GLOBAL_MAX, final_string
    # read from the filename file in /data directory
    with open(filename, "r") as f:
        data = f.read()

    # split the data into lines
    lines = data.split("\n")
    # initial pass to find min/max for normalization
    for line in lines:
        frame = [float(num.strip()) for num in line[1:-1].split(",")]
        current_min = min(
            frame[::7]
        )  # take every 7th value starting from 0th, which are x coordinates
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

    # process each line of data, add it to a string
    for line in lines:
        processed_frame = process_frame(line)

        # write the processed data to the new file in the specified format
        final_string += f"{processed_frame}\n"


# Processes a single line of text from the raw data file, applying
# Rodrigues rotation to the joint orientations


# param - frame is a single line of text from the data file
# representing all the joints of a single frame
def process_frame(frame, num_joints=32, num_features=7):
    global GLOBAL_MIN, GLOBAL_MAX, final_string
    # remove the first and last character in each line (brackets)
    frame = frame[1:-1]

    # split the frame into a list of numbers
    frame = [num.strip() for num in frame.split(",")]

    # convert each number from a string to a float
    frame = [float(num.strip()) for num in frame]

    # convert the list into a numpy array
    frame = np.array(frame)

    # reshape the array into a 2D array
    frame = frame.reshape(num_joints, num_features)

    # create a new 1-D array to hold the processed data
    processed_frame = np.zeros((num_joints * num_features))

    # loop through each joint
    for i in range(num_joints):
        # get the joint position (first three elements of each row)
        joint_position = frame[i, :3]
        # apply min-max normalization to the XYZ coordinates
        joint_position = (joint_position - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
        # get the joint orientation (last four elements of each row)
        joint_orientation = frame[i, 3:]
        rodrigues_rotation = quat_to_rodrigues(joint_orientation)

        index = i * num_features

        # positional components of final data for each joint in each frame
        processed_frame[index] = joint_position[0]  # x
        processed_frame[index + 1] = joint_position[1]  # y
        processed_frame[index + 2] = joint_position[2]  # z
        # Rodrigues rotation formula
        processed_frame[index + 3] = rodrigues_rotation[0]  # angle
        processed_frame[index + 4] = rodrigues_rotation[1][0]  # axis x
        processed_frame[index + 5] = rodrigues_rotation[1][1]  # axis y
        processed_frame[index + 6] = rodrigues_rotation[1][2]  # axis z

    # return the processed frame
    return processed_frame.tolist()


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
    def __init__(self, file_path, separator="NULL"):
        with open(file_path, "r") as f:
            lines = f.readlines()[4:-2]  # skipping the header and end notes

            # Convert lines to raw data
            self.raw_data = []
            for line in lines:
                if "NULL" in line:  # NULL separator line
                    self.raw_data.append(separator)
                else:
                    self.raw_data.append(list(map(float, line.strip()[1:-1].split(","))))

            # Split data into segments based on the separator
            self.segments = []
            start_idx = 0
            for idx, frame in enumerate(self.raw_data):
                if frame == separator:
                    self.segments.append(self.raw_data[start_idx:idx])
                    start_idx = idx + 1  # +1 to skip the separator itself
            if start_idx != len(self.raw_data):  # Append the last segment
                self.segments.append(self.raw_data[start_idx:])

    def __len__(self):
        # Calculate the valid starting indices for all segments combined
        valid_starts = [
            idx
            for segment in self.segments
            for idx in range(len(segment) - (num_frames * 2 - 1))
        ]
        return len(valid_starts)

    def __getitem__(self, index):
        # Find the appropriate segment and local index within that segment
        segment_idx, local_idx = self.global_to_local_index(index)

        # Fetch the input and target frames from the appropriate segment
        sample = self.segments[segment_idx][local_idx : local_idx + num_frames]
        all_joint_data = self.segments[segment_idx][
            local_idx + num_frames : local_idx + num_frames * 2
        ]
        target = [frame_data[:3] for frame_data in all_joint_data]

        sample = torch.tensor(sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return sample, target

    def global_to_local_index(self, global_idx):
        """
        Convert a global index (across all segments) to a local index within a specific segment.
        """
        accumulated = 0
        for seg_idx, segment in enumerate(self.segments):
            valid_starts_in_segment = len(segment) - (num_frames * 2 - 1)
            if global_idx < accumulated + valid_starts_in_segment:
                return seg_idx, global_idx - accumulated
            accumulated += valid_starts_in_segment
        raise ValueError(f"Invalid global index: {global_idx}")


def process_all_files(directory_path, output_file):
    global final_string
    files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]

    for filename in files:
        process_data(filename)
        # Add separator after processing each file
        final_string += (
            filename + "\n" + "NULL" + "\n"
        )

    with open(output_file, "w") as f:
        f.write(
            "PROCESSED TRAINING DATA: FORMAT = X, Y, Z, Angle, Axis X, Axis Y, Axis Z\n"
        )
        f.write(
            "GLOBAL_MIN: "
            + str(GLOBAL_MIN)
            + "\n"
            + "GLOBAL_MAX: "
            + str(GLOBAL_MAX)
            + "\n\n"
        )
        f.write(final_string)
        f.write("\n\nEND OF PROCESSED DATA\n\n")


if __name__ == "__main__":
    # Directory containing all raw data files
    raw_data_directory = "../data/midterm-test/"

    # Process all files in the directory
    process_all_files(raw_data_directory, "processed_test.txt")

    print(f"Data processing complete. Processed data saved to 'processed_train2.txt'.")
