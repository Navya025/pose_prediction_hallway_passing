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
int final_count = 0; 
double array[10];
typedef struct {
    float x, y;
} Point;
Point *points = NULL;

// Translation parameters
int tx = 4, ty = 5;
int translationCount = 0;

// This creates delay between two actions
void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}




void generateArray() {
    FILE *file = fopen("right-predict.txt", "r");

    char line[4096];
    Point previous = {0.0f, 0.0f}; 

    // array of points
    size_t count = 0;

    while (fgets(line, sizeof(line), file)) {
        // get x y z and coordinate of first element (pelvis)
        float x, y, z;
        if (sscanf(line, "[%f,%f,%f", &x, &y, &z) == 3) {
            // point struct to hold pelvis x, y, z of current frame
            Point current = {x, y};
            // calculate displacement from previous point
            Point displacement = {
                current.x - previous.x,
                current.y - previous.y
            };

            // add displacement to the array
            points = realloc(points, (count + 1) * sizeof(Point));
            points[count] = displacement;
            count++;

            // update previous
            previous = current;
        }
    }

    // close file
    fclose(file);
    final_count = count; 
    // printing for debugging
    int counter = 2; 
    printf("\nPoints in the array:\n");
    for (size_t i = 0; i < final_count; i++) {
        printf("%.5f, %.5f\n", points[i].x, points[i].y);
        array[counter] = points[i].x;
        array[counter + 1] = points[i].y;
        counter = counter + 2; 
    }

    // free array
    free(points);

}

// This is a basic init for the glut window
void myInit(void)
{
    glClearColor(1.0, 1.0, 1.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 300.0, 0.00, 300.0);
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

    //glClear(GL_COLOR_BUFFER_BIT);
    drawPoint();
    glFlush();
    delay(1); // Adjust the delay if needed
}

// Driver code
int main(int argc, char **argv)
{
    generateArray(); 
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
