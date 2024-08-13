using Emgu.CV;
#if USE_CUDA
using Emgu.CV.Cuda;
#endif
using Emgu.CV.CvEnum;

namespace AITrackDotnet;

internal class PositionSolver
{
    private const double ToRad = 3.14159265 / 180.0;
    private const double ToDeg = 180.0 / 3.14159265;

    private const int NbContourPointsBase = 18;

    private static readonly int[] ContourIndices = [0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45];
    private static readonly Mat LandmarkPointsBuffer = new(NbContourPointsBase, 1, DepthType.Cv32F, 2);

    private const double PriorPitch = -1.57;
    private const double PriorYaw = -1.57;
    private const double PriorDistance = -2.0;

    // Prior rotations and translation
    private static readonly double[] Rv = [0, 0, 0];
    private static readonly double[] Tv = [0, 0, 0];

    private static readonly Mat Head3dScale = new(3, 3, DepthType.Cv64F, 1);
    private static readonly Mat Mat3dContour = new(NbContourPointsBase, 3, DepthType.Cv64F, 1);
    private static readonly Mat CameraMatrix = new(3, 3, DepthType.Cv64F, 1);
    private static readonly Mat CameraDistortion = new(4, 1, DepthType.Cv64F, 1);

    public PositionSolver(int imWidth, int imHeight, float fov, float xScale, float yScale, float zScale)
    {
        Rv[0] = PriorPitch;
        Rv[1] = PriorYaw;
        Rv[2] = -1.57;
        Tv[2] = PriorDistance;

        Head3dScale.SetTo
        (
            [
                yScale, 0.0, 0.0,
                0.0, xScale, 0.0,
                0.0, 0.0, zScale
            ]
        );

        Mat3dContour.SetTo
        (
            [
                0.45517698, -0.30089578, 0.76442945,
                0.44899884, -0.16699584, 0.765143,
                0, 0.621079, 0.28729478,
                -0.44899884, -0.16699584, 0.765143,
                -0.45517698, -0.30089578, 0.76442945,
                0, -0.2933326, 0.1375821,
                0, -0.1948287, 0.06915811,
                0, -0.10384402, 0.00915182,
                0, 0, 0,
                0.08062635, 0.04127607, 0.13416104,
                0.04643935, 0.05767522, 0.10299063,
                0, 0.06875312, 0.09054535,
                -0.04643935, 0.05767522, 0.10299063,
                -0.08062635, 0.04127607, 0.13416104,
                0.31590518, -0.2983375, 0.2851074,
                0.13122973, -0.28444737, 0.23423915,
                -0.13122973, -0.28444737, 0.23423915,
                -0.31590518, -0.2983375, 0.2851074
            ]
        );

        // Taken from https://github.com/opentrack/opentrack/blob/3cc3ef246ad71c463c8952bcc96984b25d85b516/tracker-aruco/ftnoir_tracker_aruco.cpp#L193
        // Taking into account the camera FOV instead of assuming raw image dims is more clever and will make the solver more camera-agnostic.
        float diagFov = (float)(fov * ToRad);

        var widthSquared = imWidth * imWidth;
        var heightSquared = imHeight * imHeight;
        var diagonalSquared = widthSquared + heightSquared;
        var diagonal = Math.Sqrt(diagonalSquared);

        double focalLengthWidth = imWidth;
        double focalLengthHeight = imHeight;
        if (fov != 0.0)
        {
            double fovW = (double)diagFov * imWidth / diagonal;
            double fovH = (double)diagFov * imHeight / diagonal;

            focalLengthWidth = 0.5 * imWidth / Math.Tan(0.5 * fovW);
            focalLengthHeight = 0.5 * imHeight / Math.Tan(0.5 * fovH);
        }

        CameraMatrix.SetTo
        (
            [
                focalLengthHeight, 0, imHeight / 2d,
                0, focalLengthWidth, imWidth / 2d,
                0, 0, 1
            ]
        );

        CameraDistortion.SetTo([0, 0, 0, 0]);

        // This was in original code, but it seems like it's not needed

// #if USE_CUDA
//         using var mat3dContourGpu = new GpuMat(Mat3dContour);
//         using var head3dScaleGpu = new GpuMat(Head3dScale);
//         using var emptyMatGpu = new GpuMat();
//
//         CudaInvoke.Transpose(mat3dContourGpu, mat3dContourGpu);
//         CudaInvoke.Gemm(head3dScaleGpu, mat3dContourGpu, 1, emptyMatGpu, 0, mat3dContourGpu);
//         CudaInvoke.Transpose(mat3dContourGpu, mat3dContourGpu);
//
//         mat3dContourGpu.Download(Mat3dContour);
// #else
//         CvInvoke.Transpose(Mat3dContour, Mat3dContour);
//         CvInvoke.Gemm(Head3dScale, Mat3dContour, 1, null, 0, Mat3dContour);
//         CvInvoke.Transpose(Mat3dContour, Mat3dContour);
// #endif
    }

    public unsafe void SolveRotation()
    {
        var landmarkPointsBufferPtr = (float*)LandmarkPointsBuffer.GetDataPointer();

        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < ContourIndices.Length; i++)
            {
                var contourIdx = ContourIndices[i];
                landmarkPointsBufferPtr[2 * i + j] = FaceData.LandmarkCoords[2 * contourIdx + j];
            }
        }

        using var rotationVector = new Mat(Rv.Length, 1, DepthType.Cv64F, 1);
        rotationVector.SetTo(Rv);
        using var translationVector = new Mat(Tv.Length, 1, DepthType.Cv64F, 1);
        translationVector.SetTo(Tv);

        CvInvoke.SolvePnP(Mat3dContour, LandmarkPointsBuffer, CameraMatrix, CameraDistortion, rotationVector, translationVector, useExtrinsicGuess: true, flags: SolvePnpMethod.Iterative);

        GetEuler(rotationVector, translationVector);

        var rotationVectorPtr = (double*)rotationVector.GetDataPointer();
        var translationVectorPtr = (double*)translationVector.GetDataPointer();

        for (int i = 0; i < 3; i++)
        {
            FaceData.Rotation[i] = rotationVectorPtr[i];
            FaceData.Translation[i] = translationVectorPtr[i] * 10; // scale to centimeters
        }

        CorrectRotation();
        ClipRotations();
    }

    private static void GetEuler(Mat rotationVector, Mat translationVector)
    {
        using var rotMat = new Mat(3, 3, DepthType.Cv64F, 1);
        CvInvoke.Rodrigues(rotationVector, rotMat);

        using var projectionMat = new Mat(3, 4, DepthType.Cv64F, 1);
        CvInvoke.HConcat(rotMat, translationVector, projectionMat);

        using var cameraMatrix = new Mat(3, 3, DepthType.Cv64F, 1);
        using var rotMatrix = new Mat(3, 3, DepthType.Cv64F, 1);
        using var transVect = new Mat(4, 1, DepthType.Cv64F, 1);

        CvInvoke.DecomposeProjectionMatrix(
            projectionMat,
            cameraMatrix,
            rotMatrix,
            transVect,
            eulerAngles: rotationVector
        );
    }

    private static void CorrectRotation()
    {
        float distance = (float)Math.Abs(FaceData.Translation[2]);
        float lateralOffset = (float)FaceData.Translation[1];
        float verticalOffset = (float)FaceData.Translation[0];

        double correctionYaw = 90.0f - (float)Math.Atan2(distance, Math.Abs(lateralOffset)) * ToDeg;
        double correctionPitch = 90.0f - (float)Math.Atan2(distance, Math.Abs(verticalOffset)) * ToDeg;

        if (lateralOffset < 0)
            correctionYaw *= -1;

        if (verticalOffset < 0)
            correctionPitch *= -1;

        FaceData.Rotation[1] += correctionYaw;
        FaceData.Rotation[0] += correctionPitch;
    }

    private static void ClipRotations()
    {
        FaceData.Rotation[1] = FaceData.Rotation[1] switch
        {
            // Limit yaw between -90.0 and +90.0 degrees
            >= 90.0 => 90.0,
            <= -90.0 => -90.0,
            _ => FaceData.Rotation[1]
        };

        FaceData.Rotation[0] = FaceData.Rotation[0] switch
        {
            // Limit pitch between -90.0 and +90.0
            >= 90.0 => 90.0,
            <= -90.0 => -90.0,
            _ => FaceData.Rotation[0]
        };

        FaceData.Rotation[2] = FaceData.Rotation[2] switch
        {
            // Limit roll between 0.0 and +180.0
            >= 180.0 => 180.0,
            <= 0.0 => 0.0,
            _ => FaceData.Rotation[2]
        };
    }
}