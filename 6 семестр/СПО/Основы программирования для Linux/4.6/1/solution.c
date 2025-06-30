#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BUFFER_SIZE 4096

int main(int argc, char *argv[]) {
    char command[1024];
    snprintf(command, sizeof(command), "./%s %s", argv[1], argv[2]);

    FILE *fp = popen(command, "r");

    int count = 0;
    char buffer[BUFFER_SIZE];
    size_t bytesRead;

    while ((bytesRead = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        for (size_t i = 0; i < bytesRead; ++i) {
            if (buffer[i] == '0') {
                count++;
            }
        }
    }

    pclose(fp);

    printf("%d\n", count);
    return 0;
}
