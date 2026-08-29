CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -pthread
INCLUDES = -I. -I../shared
LIBS = -pthread

# Architecture-specific SIMD flags
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    CXXFLAGS += -mavx2 -mfma
endif

# If whisper.cpp is installed locally, point to its directory:
# WHISPER_DIR = /path/to/whisper.cpp
# INCLUDES += -I$(WHISPER_DIR)
# LIBS += -L$(WHISPER_DIR) -lwhisper

TARGET = whisper_server
SRCS = server_whisper.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)
