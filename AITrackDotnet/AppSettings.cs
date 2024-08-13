using Microsoft.Extensions.Configuration;
using Serilog;

namespace AITrackDotnet;

public static class AppSettings
{
    public static bool Preview { get; private set; }

    public static int FaceDetectionResizeTo { get; private set; }

    public static bool LandmarkDetectionNoiseFilter { get; private set; }

    public static bool OpenTrackUdpClientEnabled { get; private set; }
    public static string OpenTrackUdpClientHostName { get; private set; } = "localhost";
    public static int OpenTrackUdpClientPort { get; private set; }

    public static int CameraWidth { get; private set; }
    public static int CameraHeight { get; private set; }
    public static int CameraFps { get; private set; }
    public static bool CameraAutoFocus { get; private set; }

    public static bool NeedsCameraRestart { get; set; }
    public static bool WasReloaded { get; set; }

    public static void Load(IConfiguration configuration)
    {
        var previousFaceDetectionResizeTo = FaceDetectionResizeTo;
        var previousLandmarkDetectionNoiseFilter = LandmarkDetectionNoiseFilter;
        var previousCameraWidth = CameraWidth;
        var previousCameraHeight = CameraHeight;
        var previousCameraFps = CameraFps;
        var previousCameraAutoFocus = CameraAutoFocus;

#pragma warning disable IL2026 // Primitive types are safe for the trimmer
        Preview = configuration.GetValue<bool>("Preview");

        FaceDetectionResizeTo = configuration.GetValue<int>("FaceDetection:ResizeTo");

        LandmarkDetectionNoiseFilter = configuration.GetValue<bool>("LandmarkDetection:NoiseFilter");

        OpenTrackUdpClientEnabled = configuration.GetValue<bool>("OpenTrackUdpClient:Enabled");
        OpenTrackUdpClientHostName = configuration.GetValue<string>("OpenTrackUdpClient:HostName") ?? "localhost";
        OpenTrackUdpClientPort = configuration.GetValue<int>("OpenTrackUdpClient:Port");

        CameraWidth = configuration.GetValue<int>("Camera:Width");
        CameraHeight = configuration.GetValue<int>("Camera:Height");
        CameraFps = configuration.GetValue<int>("Camera:Fps");
        CameraAutoFocus = configuration.GetValue<bool>("Camera:AutoFocus");
#pragma warning restore IL2026

        if (FaceDetectionResizeTo < 84)
        {
            Log.Warning("FaceDetection:ResizeTo is too small, setting to 84");
            FaceDetectionResizeTo = 84;
        }
        else if (FaceDetectionResizeTo > 200)
        {
            Log.Warning("FaceDetection:ResizeTo is too large, this may cause performance issues, consider lowering it");
        }

        WasReloaded = true;

        NeedsCameraRestart =
            previousFaceDetectionResizeTo != FaceDetectionResizeTo ||
            previousLandmarkDetectionNoiseFilter != LandmarkDetectionNoiseFilter ||
            previousCameraWidth != CameraWidth ||
            previousCameraHeight != CameraHeight ||
            previousCameraFps != CameraFps ||
            previousCameraAutoFocus != CameraAutoFocus;
    }
}