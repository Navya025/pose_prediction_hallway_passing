#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float x, y, z;
} Point;

int main() {
    // open file
    FILE *file = fopen("helper.txt", "r");

    // Read each line from the file
    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        // get x, y, z of pelvis joint
        float x, y, z;
        if (sscanf(line, "[%f,%f,%f", &x, &y, &z) == 3) {
            // create Point struct with extracted coordinates
            Point point = {x, y, z};

            // output to check while debugging
            printf("Point: %.2f, %.2f, %.2f\n", point.x, point.y, point.z);
        } else {
            // error case if line does not have complete data
            fprintf(stderr, "Invalid line format: %s", line);
        }
    }
    fclose(file);

    return 0;
}
