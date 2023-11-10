#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float x, y;
} Point;

int main() {
    // get file
    FILE *file = fopen("helper.txt", "r");

    char line[4096];
    Point previous = {0.0f, 0.0f}; 

    // array of points
    Point *points = NULL;
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

    // printing for debugging
    printf("\nPoints in the array:\n");
    for (size_t i = 0; i < count; i++) {
        printf("%.2f, %.2f\n", points[i].x, points[i].y);
    }

    // free array
    free(points);

    return 0;
}
