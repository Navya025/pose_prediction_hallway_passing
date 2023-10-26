# Set the minimum version of CMake that can be used
# to interpret this CMakeLists file.
cmake_minimum_required(VERSION 3.10)

# Define the project name and the languages it uses.
project(calibration VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Define the source files.
# Here, we assume there's a main.cpp and some other .cpp files.
# Adjust accordingly.
set(SOURCES
    kinect_video_display.cpp
    record_calibration_data.cpp
)

# Create an executable based on the sources.
add_executable(calibration ${SOURCES})

# If you have include directories, you can specify them like this:
# include_directories(path_to_includes)

# If you have libraries to link, you can link them like this:
target_link_libraries(calibration k4a ${OpenCV_LIBS} ${GTK3_LIBRARIES} gthread-2.0)
target_include_directories(calibration PRIVATE ${OpenCV_INCLUDE_DIRS} ${GTK3_INCLUDE_DIRS})
target_compile_options(calibration PRIVATE ${GTK3_CFLAGS_OTHER})

# If you're using third-party packages or libraries,
# you might need to use 'find_package' and then link those libraries.
