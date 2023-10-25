cmake_minimum_required (VERSION 3.0)
project (visualization)

find_package(OpenGL REQUIRED)
find_package(GLEW 1.13.0 REQUIRED)
set(GLM_INCLUDE_DIR "external/glm-0.9.7.1")
set(GLM_INCLUDE_DIR "external/glew-1.13.0")

if(INCLUDE_DISTRIB)
	add_subdirectory(distrib)
endif(INCLUDE_DISTRIB)

include_directories(
	external/AntTweakBar-1.16/include/
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	external/assimp-3.0.1270/include/
	external/bullet-2.81-rev2613/src/
	.
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
	glm
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

# Tutorial 1
add_executable(joint_visualizations joint_visualizations/visualize.cpp)

target_link_libraries(joint_visualizations
	${ALL_LIBS}
	GLEW::GLEW
)
# Xcode and Visual working directories
set_target_properties(joint_visualizations PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/joint_visualizations/")

# User playground
add_executable(playground 
	playground.cpp
	common/shader.cpp
	common/shader.hpp
)
target_link_libraries(playground
	${ALL_LIBS}
)
# Xcode and Visual working directories
set_target_properties(playground PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/playground/")
create_target_launcher(playground WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/playground/")