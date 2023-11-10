#include <GL/glut.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Window size
#define maxWD 640
#define maxHT 480

// Animation parameters
int hipX = 0.2124961254108306, hipY = 0.1269369842356025, hipdX = 0.2124961254108306 - 0.19589827348497596, hipdY = 0.1269369842356025 - 0.1262360727259671, frame_count = 3;

// Point coordinates
int px = 100, py = 200;

// Translation parameters
int tx = 1, ty = 5;
int translationCount = 0;

// This creates delay between two actions
void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}

// This is a basic init for the glut window
void myInit(void)
{
    glClearColor(1.0, 1.0, 1.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, maxWD, 0.0, maxHT);
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();
}

// This function just draws a point
void drawPoint()
{
    glPointSize(7.0);
    glColor3f(1.0f, 0.0f, 1.0f);
    glBegin(GL_POINTS);
    glVertex2i(px, py);
    glEnd();
    glFlush();
}

// This function will translate the point
void translatePoint(int, int);

// Actual display function
void myDisplay()
{
    drawPoint();
    translatePoint(tx, ty);
}

// Timer function for animation
void timerFunction(int value)
{
    if (translationCount < 5)
    {
        translatePoint(tx, ty);
        glutTimerFunc(10, timerFunction, 0); // 1000 milliseconds delay
        translationCount++;
    }
}

// This function will translate the point
void translatePoint(int tx, int ty)
{
    px = px + tx;
    py = py + ty;

    // Check overflow to keep point in the screen
    if (px > maxWD || px < 0 || py > maxHT || py < 0)
    {
        px = 100; // Reset to initial position
        py = 200;
    }

    glClear(GL_COLOR_BUFFER_BIT);
    drawPoint();
    glFlush();
    delay(1); // Adjust the delay if needed
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

    // Start the timer for animation
    glutTimerFunc(1000, timerFunction, 0);

    glutMainLoop();
    return 0;
}
