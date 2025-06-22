#!/bin/bash
# Cross-platform mesh painter compilation script

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Compiling for macOS..."
    # Use clang++ on macOS
    clang++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup \
        $(python3 -m pybind11 --includes) \
        mesh_inpaint_processor.cpp \
        -o mesh_inpaint_processor$(python3-config --extension-suffix)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Compiling for Linux..."
    # Use g++ on Linux
    g++ -O3 -Wall -shared -std=c++11 -fPIC \
        $(python3 -m pybind11 --includes) \
        mesh_inpaint_processor.cpp \
        -o mesh_inpaint_processor$(python3-config --extension-suffix)
else
    echo "❓ Unknown platform, using default compilation..."
    # Fallback to original command
    c++ -O3 -Wall -shared -std=c++11 -fPIC \
        $(python3 -m pybind11 --includes) \
        mesh_inpaint_processor.cpp \
        -o mesh_inpaint_processor$(python3-config --extension-suffix)
fi

if [ $? -eq 0 ]; then
    echo "✅ Mesh painter compiled successfully"
else
    echo "❌ Mesh painter compilation failed"
    echo "   Make sure you have:"
    echo "   - C++ compiler installed (Xcode Command Line Tools on macOS)"
    echo "   - pybind11 installed (pip install pybind11)"
    echo "   - Python development headers"
    exit 1
fi