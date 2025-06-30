#include <netdb.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main(int argc, char *argv[]) {
  struct hostent *h = gethostbyname(argv[1]);
  int i = 0;
  while (h->h_addr_list[i] != NULL) {
    struct in_addr *a = (struct in_addr *) h->h_addr_list[i++];
    printf("%s\n", inet_ntoa(*a));
  }
  return 0;
}
