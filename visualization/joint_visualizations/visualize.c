#include <GL/glut.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Window size
#define maxWD 640
#define maxHT 480

// Animation parameters

// Point coordinates
double px = 81.17, py = 79.05;
double array[] = {-4.05, 2.72, -0.08, 0.90, -2.06, -0.82, -0.48, 2.53, -1.07, 1.62, -0.40, 0.03, -5.53, 2.84, -4.42, 0.75, -3.86, 2.13};

// Translation parameters
int tx = 0, ty = 1;
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
    if (translationCount < 9)
    {
        translatePoint(tx, ty);
        glutTimerFunc(1000, timerFunction, 0); // 1000 milliseconds delay
        translationCount++;
        printf("%f\n", array[tx]);
        printf("%f\n", array[ty]); 
        tx = tx + 2;
        ty = ty + 2; 
    }
}

// This function will translate the point
void translatePoint(int tx, int ty)
{
    px = px + array[tx];
    py = py + array[ty];
    // tx = tx + 2;
    // ty = ty + 2; 

    // Check overflow to keep point in the screen
    // if (px > maxWD || px < 0 || py > maxHT || py < 0)
    // {
    //     px = 100; // Reset to initial position
    //     py = 200;
    // }

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
