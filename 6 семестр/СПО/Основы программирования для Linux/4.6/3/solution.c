#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

int sig1 = 0;
int sig2 = 0;

void sigusr_handler(int signo) {
    if (signo == SIGUSR1) {
        sig1++;
    } else if (signo == SIGUSR2) {
        sig2++;
    }
}

void sigterm_handler(int signo) {
    printf("%d %d\n", sig1, sig2);
    exit(0);
}

int main(void) {
    signal(SIGUSR1, sigusr_handler);
    signal(SIGUSR2, sigusr_handler);
    signal(SIGTERM, sigterm_handler);

    while (1) {
        pause();
    }

    return 0;
}
