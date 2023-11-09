#include <GL/glut.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Window size
#define maxWD 640
#define maxHT 480

// Rotation speed
#define thetaSpeed 0.05

// Animation parameters
int hipX = 0.2124961254108306, hipY = 0.1269369842356025, hipdX = 0.2124961254108306 - 0.19589827348497596, hipdY = 0.1269369842356025 - 0.1262360727259671, frame_count = 3;

// This creates delay between
// two actions
void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}

// This is a basic init for the
// glut window
void myInit(void)
{
    glClearColor(1.0, 1.0, 1.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, maxWD, 0.0, maxHT);
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();
}

// This function just draws
// a point
void drawPoint(int x, int y)
{
    glPointSize(7.0);
    glColor3f(1.0f, 0.0f, 1.0f);
    glBegin(GL_POINTS);
    glVertex2i(x, y);
    glEnd();
    glFlush();
}

// This function will translate
// the point
// need to call a function that graphs the differnet hip points
void translatePoint(int px, int py,
                    int tx, int ty)
{
    int fx = px, fy = py;
    int counter = 5;
    while (1)
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // Update
        px = px + tx;
        py = py + ty;

        // Check overflow to keep
        // point in the screen
        if (px > maxWD || px < 0 ||
            py > maxHT || py < 0)
        {
            px = fx;
            py = fy;
        }

        // Drawing the point
        drawPoint(px, py);

        glFlush();

        // Creating a delay
        // So that the point can be noticed
        delay(10);
    }
}


// Actual display function
void myDisplay()
{
    while (frame_count)
    {
        translatePoint(hipX, hipY, hipdX, hipdY);
        frame_count--;
    }
}

// Driver code
int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(maxWD, maxHT);
    glutInitWindowPosition(100, 150);
    glutCreateWindow("Transforming point");
    glutDisplayFunc(myDisplay);
    myInit();
    glutMainLoop();
    return 0;
}
