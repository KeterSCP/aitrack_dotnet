using System.Drawing;
using System.Net.Sockets;
using Emgu.CV;
#if USE_CUDA
using Emgu.CV.Cuda;
#endif
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Microsoft.Extensions.Hosting;
using Microsoft.ML.OnnxRuntime;
using Serilog;

namespace AITrackDotnet.HostedServices;

public class MainLoopHostedService : BackgroundService
{
    private static readonly long[] OnnxTensorDimensions = [1, 3, 224, 224];
    private static readonly float[] OnnxBuffer = new float[OnnxTensorDimensions[1] * OnnxTensorDimensions[2] * OnnxTensorDimensions[3]];
    private static readonly RunOptions OnnxRunOptions = new();
    private static readonly OrtValue OnnxInputTensor = OrtValue.CreateTensorValueFromMemory(OnnxBuffer, OnnxTensorDimensions);

    private static readonly double[] UdpDatagramOfDoubles = new double[6];
    private static readonly byte[] UdpDatagramOfBytes = new byte[UdpDatagramOfDoubles.Length * sizeof(double)];

    private static readonly Dictionary<string, OrtValue> OnnxInputs = new()
    {
        ["input"] = OnnxInputTensor
    };

    private static readonly string[] OnnxOutputNames = ["output"];

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _ = Task.Run(() => ProcessFrames(stoppingToken), stoppingToken);
        return Task.CompletedTask;
    }

    private static unsafe void ProcessFrames(CancellationToken stoppingToken)
    {
        Log.Information("Main loop is running...");

#if USE_CUDA
        if (!CudaInvoke.HasCuda)
        {
            Log.Warning("There is no CUDA device available, falling back to CPU. Consider downloading CPU version of the program.");
        }

        var cudaDeviceId = CudaInvoke.GetDevice();
#endif

        using var camera = new VideoCapture(0, VideoCapture.API.DShow);

        UpdateCameraProperties(camera);

        using var positionSolver = new PositionSolver(imWidth: camera.Width, imHeight: camera.Height, fov: 56, xScale: 1.0f, yScale: 1.0f, zScale: 1.0f);

        using var mainMatCpu = new Mat();
        using var resizedMatCpu = new Mat();
        using var croppedFaceMatCpu = new Mat();
        using var resizedFaceMatCpu = new Mat();
        using var facesOutput = new Mat();

        using var udpClient = new UdpClient(AppSettings.OpenTrackUdpClientHostName, AppSettings.OpenTrackUdpClientPort);

        using var sessionOptions =
#if USE_CUDA
            SessionOptions.MakeSessionOptionWithCudaProvider(deviceId: cudaDeviceId);
#else
            // ReSharper disable once UsingStatementResourceInitialization
            new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            InterOpNumThreads = 1,
            IntraOpNumThreads = 1
        };
#endif

        using var session = new InferenceSession(@"Models\lm_f_fixed.onnx", sessionOptions);

#if USE_CUDA
        using var mainMatGpu = new GpuMat();
        using var resizedMatGpu = new GpuMat();
        using var croppedFaceMatGpu = new GpuMat();
        using var resizedFaceMatGpu = new GpuMat();
#endif

        UpdateResizeProperties(out var resizedSize, out var resizeScaleX, out var resizeScaleY, camera);
        using var faceDetector = CreateFaceDetector(resizedSize);

        while (!stoppingToken.IsCancellationRequested)
        {
            if (!camera.Read(mainMatCpu))
            {
                Log.Warning("Failed to read frame from camera.");
                continue;
            }

            FlipMat(
#if USE_CUDA
                mainMatGpu,
#endif
                mainMatCpu,
                FlipType.Horizontal
            );

            ResizeMat(
#if USE_CUDA
                mainMatGpu,
                resizedMatGpu,
#endif
                mainMatCpu,
                resizedMatCpu,
                resizedSize
            );

            faceDetector.Detect(resizedMatCpu, facesOutput);

            if (facesOutput.Rows > 0)
            {
                var facesPtr = (float*)facesOutput.DataPointer;

                // Always get one face (first)
                float faceTopLeftX = facesPtr[0];
                float faceTopLeftY = facesPtr[1];
                float faceWidth = facesPtr[2];
                float faceHeight = facesPtr[3];

                var scaledFaceTopLeftX = faceTopLeftX * resizeScaleX;
                var scaledFaceTopLeftY = faceTopLeftY * resizeScaleY;
                var scaledFaceWidth = faceWidth * resizeScaleX;
                var scaledFaceHeight = faceHeight * resizeScaleY;

                (FaceData.FaceCropTopLeft, FaceData.FaceCropBottomRight, FaceData.FaceCropSize) = GetFaceCrop(scaledFaceTopLeftX, scaledFaceTopLeftY, scaledFaceWidth, scaledFaceHeight, camera.Width, camera.Height);

                var cropPatchSize = new Size(FaceData.FaceCropBottomRight.X - FaceData.FaceCropTopLeft.X, FaceData.FaceCropBottomRight.Y - FaceData.FaceCropTopLeft.Y);
                var cropCenter = new PointF((FaceData.FaceCropTopLeft.X + FaceData.FaceCropBottomRight.X) / 2f, (FaceData.FaceCropTopLeft.Y + FaceData.FaceCropBottomRight.Y) / 2f);

                CvInvoke.GetRectSubPix(mainMatCpu, cropPatchSize, cropCenter, croppedFaceMatCpu);

                ResizeAndConvertMat(
#if USE_CUDA
                    croppedFaceMatGpu,
                    resizedFaceMatGpu,
#endif
                    croppedFaceMatCpu,
                    resizedFaceMatCpu,
                    new Size(224, 224),
                    DepthType.Cv32F
                );

                ImageProcessing.NormalizeAndTranspose(resizedFaceMatCpu, OnnxBuffer);

                using var outputs = session.Run(runOptions: OnnxRunOptions, inputs: OnnxInputs, outputNames: OnnxOutputNames);

                var outputTensor = outputs[0].GetTensorMutableDataAsSpan<float>();

                float faceCropScaleX = (float)FaceData.FaceCropSize.Width / OnnxTensorDimensions[2];
                float faceCropScaleY = (float)FaceData.FaceCropSize.Height / OnnxTensorDimensions[3];

                ProcessHeatmaps(outputTensor, FaceData.FaceCropTopLeft.X, FaceData.FaceCropTopLeft.Y, faceCropScaleX, faceCropScaleY);

                if (AppSettings.LandmarkDetectionNoiseFilter)
                {
                    EAFilter.Filter(FaceData.LandmarkCoords);
                }

                positionSolver.SolveRotation();

                if (AppSettings.OpenTrackUdpClientEnabled)
                {
                    UdpDatagramOfDoubles[0] = FaceData.Translation[1];
                    UdpDatagramOfDoubles[1] = FaceData.Translation[0];
                    UdpDatagramOfDoubles[2] = FaceData.Translation[2];
                    UdpDatagramOfDoubles[3] = FaceData.Rotation[1];
                    UdpDatagramOfDoubles[4] = FaceData.Rotation[0];
                    UdpDatagramOfDoubles[5] = FaceData.Rotation[2];

                    Buffer.BlockCopy(UdpDatagramOfDoubles, 0, UdpDatagramOfBytes, 0, UdpDatagramOfBytes.Length);

                    udpClient.Send(UdpDatagramOfBytes);
                }

                if (AppSettings.Preview)
                {
                    // // Draw landmarks
                    for (int i = 0; i < FaceData.LandmarksCount; i++)
                    {
                        CvInvoke.Circle
                        (
                            img: mainMatCpu,
                            center: new Point((int)FaceData.LandmarkCoords[2 * i + 1], (int)FaceData.LandmarkCoords[2 * i]),
                            radius: 1,
                            color: new MCvScalar(0, 255, 0),
                            thickness: 1
                        );
                    }

                    // Draw face rectangle
                    CvInvoke.Rectangle
                    (
                        img: mainMatCpu,
                        rect: new Rectangle
                        (
                            x: FaceData.FaceCropTopLeft.X,
                            y: FaceData.FaceCropTopLeft.Y,
                            width: FaceData.FaceCropSize.Width,
                            height: FaceData.FaceCropSize.Height
                        ),
                        color: new MCvScalar(255, 0, 0),
                        thickness: 1
                    );
                }
            }

            if (AppSettings.Preview)
            {
                CvInvoke.Imshow("Camera", mainMatCpu);
                CvInvoke.WaitKey(1);
            }

            if (AppSettings.WasReloaded)
            {
                if (!AppSettings.Preview)
                {
                    CvInvoke.DestroyAllWindows();
                }

                if (AppSettings.NeedsCameraRestart)
                {
                    UpdateCameraProperties(camera);
                    UpdateResizeProperties(out resizedSize, out resizeScaleX, out resizeScaleY, camera);

                    AppSettings.NeedsCameraRestart = false;
                }

                faceDetector.InputSize = resizedSize;

                AppSettings.WasReloaded = false;
            }
        }

        CvInvoke.DestroyAllWindows();
        OnnxRunOptions.Dispose();
        OnnxInputTensor.Dispose();

        Log.Information("Main loop is stopping...");
    }

    private static void UpdateResizeProperties(out Size resizedSize, out float resizeScaleX, out float resizeScaleY, VideoCapture cam)
    {
        var resizedWidth = AppSettings.FaceDetectionResizeTo;
        var aspectRatio = (float)cam.Width / cam.Height;
        var resizedHeight = (int)(resizedWidth / aspectRatio);

        resizedSize = new Size(resizedWidth, resizedHeight);
        resizeScaleX = (float)cam.Width / resizedWidth;
        resizeScaleY = (float)cam.Height / resizedHeight;
    }

    private static void UpdateCameraProperties(VideoCapture cam)
    {
        cam.Set(CapProp.Autofocus, AppSettings.CameraAutoFocus ? 1 : 0);
        cam.Set(CapProp.Fps, AppSettings.CameraFps);
        cam.Set(CapProp.FrameWidth, AppSettings.CameraWidth);
        cam.Set(CapProp.FrameHeight, AppSettings.CameraHeight);
    }

    private static FaceDetectorYN CreateFaceDetector(Size inputSize)
    {
        return new FaceDetectorYN(
            model: @"Models\face_detection.onnx",
            config: "",
            inputSize: inputSize,
            scoreThreshold: 0.8f,
            nmsThreshold: 0.5f,
            topK: 1,
#if USE_CUDA
            backendId: Emgu.CV.Dnn.Backend.Cuda,
            targetId: Target.Cuda

#else
            backendId: Emgu.CV.Dnn.Backend.Default,
            targetId: Target.Cpu
#endif
        );
    }

    private static void FlipMat(
#if USE_CUDA
        GpuMat srcGpu,
#endif
        Mat srcCpu,
        FlipType flipType)
    {
#if USE_CUDA
        srcGpu.Upload(srcCpu);
        CudaInvoke.Flip(srcGpu, srcGpu, flipType);
        srcGpu.Download(srcCpu);
#else
        CvInvoke.Flip(srcCpu, srcCpu, flipType);
#endif
    }

    private static void ResizeMat(
#if USE_CUDA
        GpuMat srcGpu,
        GpuMat dstGpu,
#endif
        Mat srcCpu,
        Mat dstCpu,
        Size size)
    {
#if USE_CUDA
        srcGpu.Upload(srcCpu);
        CudaInvoke.Resize(srcGpu, dstGpu, size);
        dstGpu.Download(dstCpu);
#else
        CvInvoke.Resize(srcCpu, dstCpu, size);
#endif
    }

    private static void ResizeAndConvertMat(
#if USE_CUDA
        GpuMat srcGpu,
        GpuMat dstGpu,
#endif
        Mat srcCpu,
        Mat dstCpu,
        Size size,
        DepthType depth)
    {
#if USE_CUDA
        srcGpu.Upload(srcCpu);
        CudaInvoke.Resize(srcGpu, dstGpu, size);
        dstGpu.ConvertTo(dstGpu, depth);
        dstGpu.Download(dstCpu);
#else
        CvInvoke.Resize(srcCpu, dstCpu, size);
        dstCpu.ConvertTo(dstCpu, depth);
#endif
    }

    private static (Point TopLeft, Point BottomRight, Size Size) GetFaceCrop(float scaledFaceTopLeftX, float scaledFaceTopLeftY, float scaledFaceWidth, float scaledFaceHeight, float cameraWidth, float cameraHeight)
    {
        // Force a little wider bounding box so the chin tends to be covered - 10% wider
        int cropX1 = (int)(scaledFaceTopLeftX - scaledFaceWidth * 0.1);
        int cropY1 = (int)(scaledFaceTopLeftY - scaledFaceHeight * 0.1);
        int cropX2 = (int)(scaledFaceTopLeftX + scaledFaceWidth + scaledFaceWidth * 0.1);
        int cropY2 = (int)(scaledFaceTopLeftY + scaledFaceHeight + scaledFaceHeight * 0.1f);

        var topLeftX = Math.Max(0, cropX1);
        var topLeftY = Math.Max(0, cropY1);
        var bottomRightX = Math.Min((int)cameraWidth, cropX2);
        var bottomRightY = Math.Min((int)cameraHeight, cropY2);

        var cropWidth = bottomRightX - topLeftX;
        var cropHeight = bottomRightY - topLeftY;

        return (new Point(topLeftX, topLeftY), new Point(bottomRightX, bottomRightY), new Size(cropWidth, cropHeight));
    }

    private static void ProcessHeatmaps(Span<float> heatmaps, int faceCropTopLeftX, int faceCropTopLeftY, float faceCropScaleX, float scaleY)
    {
        const int heatmapSize = 784; // 28 * 28;

        for (int landmark = 0; landmark < FaceData.LandmarksCount; landmark++)
        {
            int offset = heatmapSize * landmark;
            int argMax = -100;
            float maxVal = -100;

            var landmarkHeatmap = heatmaps[offset..(offset + heatmapSize)];
            for (int i = 0; i < heatmapSize; i++)
            {
                if (landmarkHeatmap[i] > maxVal)
                {
                    argMax = i;
                    maxVal = landmarkHeatmap[i];
                }
            }

            int x = argMax / 28;
            int y = argMax % 28;

            const float res = 223;

            var offX = (int)MathF.Floor(res * (Logit(heatmaps[66 * heatmapSize + offset + argMax])) + 0.1f);
            var offY = (int)MathF.Floor(res * (Logit(heatmaps[2 * 66 * heatmapSize + offset + argMax])) + 0.1f);

            float lmX = faceCropTopLeftY + scaleY * (res * (x / 27.0f) + offX);
            float lmY = faceCropTopLeftX + faceCropScaleX * (res * (y / 27.0f) + offY);

            FaceData.LandmarkCoords[2 * landmark] = lmX;
            FaceData.LandmarkCoords[2 * landmark + 1] = lmY;
        }
    }

    private static float Logit(float p)
    {
        if (p >= 0.9999999f)
            p = 0.9999999f;
        else if (p <= 0.0000001f)
            p = 0.0000001f;

        p /= (1.0f - p);
        return MathF.Log(p) / 16.0f;
    }
}