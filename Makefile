NVCC = nvcc
TARGET = signal_process
SRC = src/main.cu

# Google Style generally implies strict warnings, but we keep it simple for compatibility
FLAGS = -O2

all:
	$(NVCC) $(FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
	rm -f data/output/*