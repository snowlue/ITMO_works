#include <stdio.h>

int main() {
    FILE *fp = fopen("/proc/self/stat", "r");
    if (fp == NULL) {
        perror("fopen");
        return 1;
    }

    int pid, ppid;
    char comm[256];
    char state;

    if (fscanf(fp, "%d %255s %c %d", &pid, comm, &state, &ppid) != 4) {
        perror("fscanf");
        fclose(fp);
        return 1;
    }

    fclose(fp);

    printf("%d\n", ppid);
    return 0;
}
