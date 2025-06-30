#include <fcntl.h>
#include <stdio.h>
#include <sys/select.h>
#include <unistd.h>

int main(void) {
    int rc, maxfd, fd[2], sum[2] = {0, 0};
    struct timeval tm = {5, 0};
    char buff[32];

    fd_set watchset;
    fd_set inset;

    fd[0] = open("in1", O_RDONLY | O_NONBLOCK);
    fd[1] = open("in2", O_RDONLY | O_NONBLOCK);

    FD_ZERO(&watchset);
    FD_SET(fd[0], &watchset);
    FD_SET(fd[1], &watchset);

    maxfd = fd[0] > fd[1] ? fd[0] : fd[1];
    while (FD_ISSET(fd[0], &watchset) || FD_ISSET(fd[1], &watchset)) {
        inset = watchset;
        if (select(maxfd + 1, &inset, NULL, NULL, &tm) < 0) {
            perror("select");
            return 1;
        }

        for (int i = 0; i < 2; i++) {
            if (FD_ISSET(fd[i], &inset)) {
                rc = read(fd[i], buff, sizeof(char));
                if (rc < 0) {
                    perror("read");
                    return 1;
                } else if (!rc) {
                    close(fd[i]);
                    FD_CLR(fd[i], &watchset);
                } else {
                    buff[rc] = '\0';
                    sscanf(buff, "%d", &rc);
                    sum[i] += rc;
                }
            }
        }
    }
    printf("%d\n", sum[0] + sum[1]);
    return 0;
}
