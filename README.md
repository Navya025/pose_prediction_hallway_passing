# pose_prediction_hallway_passing

### Step 1: Set Up
Download and Set Up Azure Kinect SDK: [Azure Kinect SDK Download Instructions](https://learn.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download)

Download and Set up Body Tracking SDK: [Body Tracking SDK Download Instructions](https://learn.microsoft.com/en-us/azure/kinect-dk/body-sdk-setup)

For the last step use: sudo apt-get install -y libk4a1.4 libk4a1.4-dev k4a-tools

### Step 2: Verify Download
Verify the download by running k4aviewer. This will open the Azure Kinect, to start the cameria click "Open Device" and then "Start Camera".

To run body tracking use the command k4abt_simple_3d_viewer.

### Step 3: Run joint_recognition_code
Run the code (after you cd into the joint_recognition folder) with this command: g++ main.cpp -Wall -I/home/bwilab/hallway_lane_detection_fri2/Azure-Kinect-Sensor-SDK -L/home/bwilab/hallway_lane_detection_fri2/Azure-Kinect-Sensor-SDK -lk4a -lk4abt -o a.out

This will create an output file in the same directory, to view the data run ./[insert name of output file], for example if the output file is a.out, run ./a.out.

### Run Visualization code
Run the code (after you cd in visualization/joint_visualization):gcc -o visualize visualize.c -lGL -lGLU -lglut -lm 

This will create an output file which you can run by running: ./visualize



