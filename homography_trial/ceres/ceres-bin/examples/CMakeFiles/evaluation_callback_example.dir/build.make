# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/bwilab/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/bwilab/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin

# Include any dependencies generated for this target.
include examples/CMakeFiles/evaluation_callback_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/evaluation_callback_example.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/evaluation_callback_example.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/evaluation_callback_example.dir/flags.make

examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o: examples/CMakeFiles/evaluation_callback_example.dir/flags.make
examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o: /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0/examples/evaluation_callback_example.cc
examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o: examples/CMakeFiles/evaluation_callback_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o"
	cd /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o -MF CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o.d -o CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o -c /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0/examples/evaluation_callback_example.cc

examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.i"
	cd /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0/examples/evaluation_callback_example.cc > CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.i

examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.s"
	cd /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0/examples/evaluation_callback_example.cc -o CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.s

# Object files for target evaluation_callback_example
evaluation_callback_example_OBJECTS = \
"CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o"

# External object files for target evaluation_callback_example
evaluation_callback_example_EXTERNAL_OBJECTS =

bin/evaluation_callback_example: examples/CMakeFiles/evaluation_callback_example.dir/evaluation_callback_example.cc.o
bin/evaluation_callback_example: examples/CMakeFiles/evaluation_callback_example.dir/build.make
bin/evaluation_callback_example: lib/libceres.a
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libglog.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libamd.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libtbb.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcudart.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcusolver.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcublas.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libculibos.a
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libcusparse.so
bin/evaluation_callback_example: lib/libceres_cuda_kernels.a
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libblas.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/evaluation_callback_example: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/evaluation_callback_example: examples/CMakeFiles/evaluation_callback_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/evaluation_callback_example"
	cd /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/evaluation_callback_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/evaluation_callback_example.dir/build: bin/evaluation_callback_example
.PHONY : examples/CMakeFiles/evaluation_callback_example.dir/build

examples/CMakeFiles/evaluation_callback_example.dir/clean:
	cd /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples && $(CMAKE_COMMAND) -P CMakeFiles/evaluation_callback_example.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/evaluation_callback_example.dir/clean

examples/CMakeFiles/evaluation_callback_example.dir/depend:
	cd /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0 /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-solver-2.2.0/examples /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples /home/bwilab/hallway_lane_detection_fri2/pose_prediction_hallway_passing/homography_trial/ceres/ceres-bin/examples/CMakeFiles/evaluation_callback_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/evaluation_callback_example.dir/depend
