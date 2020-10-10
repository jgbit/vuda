CC=g++
CFLAGS = -std=c++17 -Wall -g0
LDFLAGS = -lvulkan -lpthread

NVCC=nvcc
CFLAGS_NVCC = -std=c++11
LDFLAGS_NVCC = -lpthread

INCLUDE = -I../../inc/