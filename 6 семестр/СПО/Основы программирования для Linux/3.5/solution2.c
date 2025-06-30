#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    struct dirent **namelist;
    int n;

    n = scandir("/proc", &namelist, NULL, alphasort);
    if (n == -1) {
        perror("scansur");
        exit(EXIT_FAILURE);
    }

    FILE *f;
    char *name, path[1024];
    int count = 0;

    while (n--) {
        if (atoi(namelist[n]->d_name) != 0) {
            sprintf(path, "/proc/%s/comm", namelist[n]->d_name);
            f = fopen(path, "rb");

            fscanf(f, "%s", name);
            if (strcmp("genenv", name) == 0) {
                count++;
            }
            fclose(f);
        }
        free(namelist[n]);
    }
    free(namelist);
    printf("%d\n", count);

    exit(EXIT_SUCCESS);
}
