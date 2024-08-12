using System.Drawing;
using Emgu.CV;
#if USE_CUDA
using Emgu.CV.Cuda;
#endif
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Models;
using Emgu.CV.Structure;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace AITrackDotnet.HostedServices;

public class MainLoopHostedService : BackgroundService
{
    private readonly ILogger<MainLoopHostedService> _logger;

    // TODO: make this configurable
    private const bool Preview = true;
    private const bool UseCuda =
#if USE_CUDA
        true;
#else
        false;
#endif

    public MainLoopHostedService(ILogger<MainLoopHostedService> logger)
    {
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Main loop is running...");

        using var camera = new VideoCapture(0, VideoCapture.API.DShow);
        // using var facemark = new FacemarkLBF(new FacemarkLBFParams());

        camera.Set(CapProp.Autofocus, 1);
        camera.Set(CapProp.Fps, 30);
        camera.Set(CapProp.FrameWidth, 640);
        camera.Set(CapProp.FrameHeight, 480);

        float aspectRatio = (float)camera.Width / camera.Height;

        const int resizedWidth = 124;
        var resizedHeight = (int)(resizedWidth / aspectRatio);

        var resizedSize = new Size(resizedWidth, resizedHeight);

        // facemark.LoadModel("D:\\Downloads\\lbfmodel.yaml");

        var resizeScaleX = (float)camera.Width / resizedWidth;
        var resizeScaleY = (float)camera.Height / resizedHeight;

        using var mainMatCpu = new Mat();
        using var resizedMatCpu = new Mat();

#if USE_CUDA
        using var mainMatGpu = new GpuMat();
        using var resizedMatGpu = new GpuMat();
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

            using (var facesOutput = new Mat())
            {
                var detected = faceDetector.Detect(resizedMatCpu, facesOutput);

                if (detected == 1)
                {
                    var res = FaceDetectorYNModel.ConvertMatToFaceDetectorYNResult(facesOutput);

                    foreach (var face in res)
                    {
                        if (Preview)
                        {
                            CvInvoke.Rectangle
                            (
                                img: mainMatCpu,
                                rect: new Rectangle
                                (
                                    x: (int)(face.Region.X * resizeScaleX),
                                    y: (int)(face.Region.Y * resizeScaleY),
                                    width: (int)(face.Region.Width * resizeScaleX),
                                    height: (int)(face.Region.Height * resizeScaleY)
                                ),
                                color: new MCvScalar(255, 0, 0),
                                thickness: 2
                            );
                        }
                    }
                }
            }

            if (Preview)
            {
                CvInvoke.Imshow("Camera", mainMatCpu);
                CvInvoke.WaitKey(1);
            }
        }

        CvInvoke.DestroyAllWindows();
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
}