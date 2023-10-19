
add_executable(joint_detection main.cpp)


# Dependencies of this library
target_link_libraries(joint_detection PRIVATE 
    k4a
    k4abtsi
    )

