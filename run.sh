#!/bin/bash

# 1. Setup Directories
echo "Setting up directories..."
mkdir -p data/input
mkdir -p data/output

# 2. Generate Dummy Data (100 small files to meet "100s of inputs" requirement)
echo "Generating 100 synthetic signal files..."
for i in {1..100}
do
   # Create a file with random binary data
   dd if=/dev/urandom of=data/input/signal_$i.bin bs=1024 count=1 status=none
done

# 3. Compile
echo "Compiling..."
make

# 4. Execute
if [ -f "./signal_process" ]; then
    echo "Running CUDA Processing..."
    # CLI arguments passed here
    ./signal_process --input ./data/input --output ./data/output --scale 5.0
else
    echo "Compilation Failed!"
    exit 1
fi

echo "Done. Verified output count:"
ls data/output | wc -l