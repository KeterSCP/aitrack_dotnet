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
- [x] Dev environment setup + build instructions (download Emgu.CV package from custom source)
- [ ] Check if it works on Linux and MacOS
- [ ] Publish scripts and instructions

## Local development setup

* Download and install .NET >=8 SDK: https://dotnet.microsoft.com/en-us/download/dotnet/8.0
* Clone the repository
* Run the project in your IDE, or via command line with `dotnet run` (you can specify build configuration by adding `-c Release` or `-c Debug` flag)
  * You can test CUDA version, by uncommenting the `<DefineConstants>USE_CUDA</DefineConstants>` line in `.csproj` file, or by specifying it in the command line: `dotnet run -c Release /p:DefineConstants=USE_CUDA`
  * NOTE: if you are using CUDA, the first time you run/publish the project, it will take a while to load necessary Emgu.CV CUDA runtime packages, so be patient