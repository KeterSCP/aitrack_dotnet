using System.Drawing;

namespace AITrackDotnet;

internal static class FaceData
{
    public const int LandmarksCount = 66;

    public static Point FaceCropTopLeft { get; set; }
    public static Point FaceCropBottomRight { get; set; }
    public static Size FaceCropSize { get; set; }

    // Landmark positions: [[x,y], [x,y], [x,y], ...]
    public static float[] LandmarkCoords { get; } = new float[LandmarksCount * 2];

    // Rotation: [x, y, z]
    public static double[] Rotation { get; } = new double[3];
    // Translation: [x, y, z]
    public static double[] Translation { get; } = new double[3];
}