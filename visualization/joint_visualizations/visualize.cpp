#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
using namespace glm;

struct Joint {
    float x;
    float y;
    float z;
};

struct Bone {
    Joint startJoint;
    Joint endJoint;
};

struct Body {
    Joint joints[29];
    Bone bones[25];
};

void initialize_bones(Body& human) {
    // Initialize bones based on joint coordinates
    // You should populate the bone coordinates as needed
    // Example:
    human.bones[0].startJoint = human.joints[0];
    human.bones[0].endJoint = human.joints[1];
    // Repeat for other bones
}

void initialize_joints(Body& human, const float* output) {
    // Populate joint coordinates from the 'output' array
    for (int i = 1; i < 29; i++) {
        human.joints[i].x = output[i * 3];
        human.joints[i].y = output[i * 3 + 1];
        human.joints[i].z = output[i * 3 + 2];
    }
}

int main(void) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1024, 768, "Pose Prediction Joint Visualization", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    Body human; // Create a Body structure to hold joint and bone data
    float output[87]; // Replace with your actual joint and bone coordinates

    initialize_joints(human, output); // Initialize joint coordinates
    initialize_bones(human); // Initialize bones based on joint coordinates

    do {
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_LINES); // Use GL_LINES to draw lines

        // Set the line color to white (R=1, G=1, B=1)
        glColor3f(1.0f, 1.0f, 1.0f);
		
        for (int i = 0; i < 25; i++) {
            glVertex3f(human.bones[i].startJoint.x, human.bones[i].startJoint.y, human.bones[i].startJoint.z);
            glVertex3f(human.bones[i].endJoint.x, human.bones[i].endJoint.y, human.bones[i].endJoint.z);
        }

        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();

    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);

    glfwTerminate();

    return 0;
}
