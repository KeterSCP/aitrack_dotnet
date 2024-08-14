#!/bin/bash

initial_dir=$(pwd)
cd AITrackDotnet

if [ "$1" == "win-x64" ]; then
    if [ "$2" == "cuda" ]; then
        dotnet publish -c Release -r win-x64 -p:DefineConstants=USE_CUDA -o "$initial_dir/artifacts/publish_cuda"
    else
        dotnet publish -c Release -r win-x64 -o "$initial_dir/artifacts/publish_cpu"
    fi
else
    echo "Usage: ./publish.sh <runtime> [cuda]"
    echo "Example: ./publish.sh win-x64 [cuda]"
fi