#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/fs.h>

int main(void) {
    pid_t pid = fork();

    if (pid != 0)
        exit(EXIT_SUCCESS);

    if (setsid() == -1)
        return -1;

    if (chdir("/") == -1)
        return -1;

    printf("%d\n", pid);

    close(0);
    close(1);
    close(2);

    sleep(1000);
    return 0;
}
