#include <stdio.h>
#include <stdlib.h>

int get_ppid(int pid) {
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    int ppid = -1;
    int dummy;
    char comm[256];
    char state;

    if (fscanf(fp, "%d %255s %c %d", &dummy, comm, &state, &ppid) != 4) {
        ppid = -1;
    }

    fclose(fp);
    return ppid;
}

int main(int argc, char *argv[]) {
    int pid = atoi(argv[1]);

    while (pid > 0) {
        printf("%d\n", pid);
        if (pid == 1) {
            break;
        }
        pid = get_ppid(pid);
        if (pid == -1) {
            break;
        }
    }

    return 0;
}
