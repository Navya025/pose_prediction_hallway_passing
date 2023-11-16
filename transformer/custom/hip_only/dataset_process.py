#This file preprocesses the dataset outlined in https://www.sciencedirect.com/science/article/pii/S2352340923004523
#For use in training a transformer model which predicts pose based on previous joint pose data

#The dataset is organized as follows:
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



