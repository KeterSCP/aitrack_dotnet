using System.Drawing;
using Emgu.CV;
#if USE_CUDA
using Emgu.CV.Cuda;
#endif
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace AITrackDotnet.HostedServices;

public class MainLoopHostedService : BackgroundService
{
    private readonly ILogger<MainLoopHostedService> _logger;
    private static readonly long[] _onnxTensorDimensions = [1, 3, 224, 224];
    private static readonly float[] _onnxBuffer = new float[_onnxTensorDimensions[1] * _onnxTensorDimensions[2] * _onnxTensorDimensions[3]];
    private static readonly RunOptions _onnxRunOptions = new();
    private static readonly OrtValue _onnxInputTensor = OrtValue.CreateTensorValueFromMemory(_onnxBuffer, _onnxTensorDimensions);
    private static readonly Dictionary<string, OrtValue> _onnxInputs = new()
    {
        ["input"] = _onnxInputTensor
    };

    // Landmark positions: [[x,y], [x,y], [x,y], ...]
    private static readonly float[] _landmarkCoords = new float[66 * 2];

    // TODO: make this configurable
    private const bool Preview = true;
    private const bool UseCuda =
#if USE_CUDA
        true;
#else
        false;
#endif

    private static readonly string[] OnnxOutputNames = ["output"];

    public MainLoopHostedService(ILogger<MainLoopHostedService> logger)
    {
        _logger = logger;
    }

    protected override unsafe Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Main loop is running...");

        using var camera = new VideoCapture(0, VideoCapture.API.DShow);

        camera.Set(CapProp.Autofocus, 1);
        camera.Set(CapProp.Fps, 30);
        camera.Set(CapProp.FrameWidth, 640);
        camera.Set(CapProp.FrameHeight, 480);

        float aspectRatio = (float)camera.Width / camera.Height;

        const int resizedWidth = 114;
        var resizedHeight = (int)(resizedWidth / aspectRatio);

        var resizedSize = new Size(resizedWidth, resizedHeight);

        using var mainMatCpu = new Mat();
        using var resizedMatCpu = new Mat();
        using var croppedFaceMatCpu = new Mat();
        using var resizedFaceMatCpu = new Mat();
        using var facesOutput = new Mat();

        var resizeScaleX = (float)camera.Width / resizedWidth;
        var resizeScaleY = (float)camera.Height / resizedHeight;

        using var session = new InferenceSession(@"models\lm_f.onnx", new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            InterOpNumThreads = 1,
            IntraOpNumThreads = 1
        });

#if USE_CUDA
        using var mainMatGpu = new GpuMat();
        using var resizedMatGpu = new GpuMat();
        using var croppedFaceMatGpu = new GpuMat();
        using var resizedFaceMatGpu = new GpuMat();
#endif

        using var faceDetector = CreateFaceDetector(UseCuda, resizedSize);

        while (!stoppingToken.IsCancellationRequested)
        {
            if (!camera.Read(mainMatCpu))
            {
                _logger.LogWarning("Failed to read frame from camera.");
                continue;
            }

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
                float x0 = facesPtr[0];
                float y0 = facesPtr[1];
                float faceWidth = facesPtr[2];
                float faceHeight = facesPtr[3];

                var faceRectX = x0 * resizeScaleX;
                var faceRectY = y0 * resizeScaleY;
                var faceRectWidth = faceWidth * resizeScaleX;
                var faceRectHeight = faceHeight * resizeScaleY;

                ProcFaceDetect(ref faceRectX, ref faceRectY, ref faceRectWidth, ref faceRectHeight, camera.Width, camera.Height);

                var cropPatchSize = new Size((int)faceRectWidth - (int)faceRectX, (int)faceRectHeight - (int)faceRectY);
                var cropCenter = new PointF((faceRectX + faceRectWidth) / 2, (faceRectY + faceRectHeight) / 2);

                CvInvoke.GetRectSubPix(mainMatCpu, cropPatchSize, cropCenter, croppedFaceMatCpu);

                ResizeAndConvert(
#if USE_CUDA
                    croppedFaceMatGpu,
                    resizedFaceMatGpu,
#endif
                    croppedFaceMatCpu,
                    resizedFaceMatCpu,
                    new Size(224, 224),
                    DepthType.Cv32F
                );

                ImageProcessing.NormalizeAndTranspose(resizedFaceMatCpu, _onnxBuffer);

                using var outputs = session.Run(runOptions: _onnxRunOptions, inputs: _onnxInputs, outputNames: OnnxOutputNames);

                var outputTensor = outputs[0].GetTensorMutableDataAsSpan<float>();

                int width = (int)faceRectWidth - (int)faceRectX;
                int height = (int)faceRectHeight - (int)faceRectY;

                float scaleX = (float)width / _onnxTensorDimensions[2];
                float scaleY = (float)height / _onnxTensorDimensions[3];

                ProcHeatmaps(outputTensor, (int)faceRectX, (int)faceRectY, scaleX, scaleY);

                if (Preview)
                {
                    // // Draw landmarks
                    for (int i = 0; i < 66; i++)
                    {
                        CvInvoke.Circle
                        (
                            img: mainMatCpu,
                            center: new Point((int)_landmarkCoords[2 * i + 1], (int)_landmarkCoords[2 * i]),
                            radius: 2,
                            color: new MCvScalar(0, 255, 0),
                            thickness: 2
                        );
                    }

                    // Draw face rectangle
                    CvInvoke.Rectangle
                    (
                        img: mainMatCpu,
                        rect: new Rectangle
                        (
                            x: (int)faceRectX,
                            y: (int)faceRectY,
                            width: width,
                            height: height
                        ),
                        color: new MCvScalar(255, 0, 0),
                        thickness: 2
                    );
                }
            }

            if (Preview)
            {
                CvInvoke.Imshow("Camera", mainMatCpu);
                CvInvoke.WaitKey(1);
            }
        }

        CvInvoke.DestroyAllWindows();
        _onnxRunOptions.Dispose();
        _onnxInputTensor.Dispose();
        return Task.CompletedTask;
    }

    private static FaceDetectorYN CreateFaceDetector(bool useCuda, Size inputSize)
    {
        return new FaceDetectorYN(
            model: @"Models\face_detection.onnx",
            config: "",
            inputSize: inputSize,
            scoreThreshold: 0.8f,
            nmsThreshold: 0.5f,
            topK: 1,
            backendId: useCuda ? Emgu.CV.Dnn.Backend.Cuda : Emgu.CV.Dnn.Backend.Default,
            targetId: useCuda ? Target.Cuda : Target.Cpu
        );
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

    private static void ResizeAndConvert(
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

    private static void ProcFaceDetect(ref float x0, ref float y0, ref float faceWidth, ref float faceHeight, float width, float height)
    {
        int cropX1 = (int)(x0 - faceWidth * 0.1);
        int cropY1 = (int)(y0 - faceHeight * 0.1);
        int cropX2 = (int)(x0 + faceWidth + faceWidth * 0.1);
        int cropY2 = (int)(y0 + faceHeight + faceHeight * 0.1f); // force a little taller BB so the chin tends to be covered

        x0 = Math.Max(0, cropX1);
        y0 = Math.Max(0, cropY1);
        faceWidth = Math.Min((int)width, cropX2);
        faceHeight = Math.Min((int)height, cropY2);
    }

    private static void ProcHeatmaps(Span<float> heatmaps, int x0, int y0, float scaleX, float scaleY)
    {
        const int heatmapSize = 784; // 28 * 28;

        for (int landmark = 0; landmark < 66; landmark++)
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

            float lmX = y0 + scaleY * (res * (x / 27.0f) + offX);
            float lmY = x0 + scaleX * (res * (y / 27.0f) + offY);

            _landmarkCoords[2 * landmark] = lmX;
            _landmarkCoords[2 * landmark + 1] = lmY;
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