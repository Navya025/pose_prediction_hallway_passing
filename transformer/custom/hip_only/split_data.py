import os
import random

def split_data(src_file, train_file, val_file, test_file, classes, train_split=0.7, val_split=0.15, test_split=0.15):
    # Collect all segments and categorize them by classes
    class_segments = {cls: [] for cls in classes}
    with open(src_file, 'r') as f:
        lines = f.readlines()

    current_segment = []
    current_class = None
    skip_initial_metadata = 4
    skip_final_metadata = 2
    for line_number, line in enumerate(lines):
        # Skip the metadata at the beginning and end of the file
        if line_number < skip_initial_metadata or line_number >= len(lines) - skip_final_metadata:
            continue

        if 'NULL' in line:
            if current_segment and current_class:
                class_segments[current_class].append(current_segment)
            current_segment = []
        elif any(cls in line for cls in classes):
            current_class = next(cls for cls in classes if cls in line)
        else:
            current_segment.append(line)

    # Add the last segment if it's valid
    if current_segment and current_class:
        class_segments[current_class].append(current_segment)

    # Shuffle and split data for each class
    train_segments, val_segments, test_segments = [], [], []
    for cls, segments in class_segments.items():
            random.shuffle(segments)
            total = len(segments)
            train_count = int(total * train_split)
            val_count = int(total * val_split)
            test_count = int(total * test_split)  # Explicitly calculate test_count

            # Adjust if total is not perfectly divisible
            remaining = total - (train_count + val_count + test_count)
            if remaining > 0:
                train_count += remaining  # Add the remainder to the train set

            train_segments.extend(segments[:train_count])
            val_segments.extend(segments[train_count:train_count + val_count])
            test_segments.extend(segments[train_count + val_count:train_count + val_count + test_count])


    # Shuffle the combined segments
    random.shuffle(train_segments)
    random.shuffle(val_segments)
    random.shuffle(test_segments)

    # Write segments to their respective files
    for segments, output_file in [(train_segments, train_file), (val_segments, val_file), (test_segments, test_file)]:
        with open(output_file, 'w') as f:
            for segment in segments:
                f.writelines(segment)
                f.write('NULL\n')

    print("Successful split!")

# File paths
src_file = "processed_train_final.txt"
train_file = "2FINAL_train_data.txt"
val_file = "2FINAL_val_data.txt"
test_file = "2FINAL_test_data.txt"
classes = ["_s.txt", "_ro.txt", "_rog.txt", "_lo.txt", "_log.txt"]

# Perform the split
split_data(src_file, train_file, val_file, test_file, classes)
