#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PATH_PATTERN \
"/proc/%d/task/%d/children"
#define SIZE 1024

char path[SIZE];

int dfs(int pid) {
    int res = 1;
    sprintf(path, PATH_PATTERN, pid, pid);
    FILE *f = fopen(path, "rb");
    int id;
    while (fscanf(f, "%d", &id) >= 0) {
        res += dfs(id);
    }
    fclose(f);
    return res;
}

int main(int argc, char *argv[]) {
    int pid = atoi(argv[1]);
    printf("%d\n", dfs(pid));
    return 0;
}
