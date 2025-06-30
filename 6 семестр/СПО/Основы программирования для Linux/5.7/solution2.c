#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <arpa/inet.h>

struct sockaddr_in local;

int main(int argc, char *argv[]) {
    int s = socket(AF_INET, SOCK_DGRAM, 0);

    inet_aton("127.0.0.1", &local.sin_addr);
    local.sin_port = htons(atoi(argv[1]));
    local.sin_family = AF_INET;

    int result = bind(s, (struct sockaddr *) &local, sizeof(local));

    char buff[BUFSIZ];
    const char *target = "OFF\n";
    while (1) {
        size_t len = read(s, buff, BUFSIZ);
        buff[len] = '\0';
        if (strcmp(buff, target) == 0) {
            break;
        }
        printf("%s\n", buff);
    }
    return 0;
}
