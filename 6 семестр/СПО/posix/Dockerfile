FROM ubuntu:18.04
LABEL name="posix"

RUN apt update 
RUN apt install -y --allow-unauthenticated \
    clang \
    g++ \
    clang-tidy \
    clang-format-7 \
    make

RUN ln -s /usr/bin/clang-format-7 /usr/bin/clang-format
