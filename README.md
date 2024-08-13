# AITrackDotnet

This project is a port of the [AI Track](https://github.com/AIRLegend/aitrack) project to .NET.

It aims to be cross-platform with possibility to use CUDA (GPU) for faster processing.

## Changes from the original project

- No GUI for configuration - everything can be configured (with live reloading) via a JSON file appsettings.json
- Only 1 model is used for landmarks detection, originally known as "Fast" (lm_f.onnx)
- Support for PS3 camera is not included, only WebCam is supported

## Current status

- On windows both CUDA and CPU versions work, and correctly send data to the OpenTrack via UDP

## TODO

- [ ] Refactor and structure the code
- [ ] Verify places that could be processed by CUDA
- [ ] Dev environment setup + build instructions (download Emgu.CV package from custom source)
- [ ] Check if it works on Linux and MacOS
- [ ] Publish scripts