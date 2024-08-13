namespace AITrackDotnet;

internal static class EAFilter
{
    private static readonly float[] LastValue = new float[FaceData.LandmarksCount * 2];

    public static void Filter(float[] landmarks)
    {
        for (int i = 0; i < landmarks.Length; i++)
        {
            landmarks[i] = 0.6f * landmarks[i] + 0.4f * LastValue[i];
            LastValue[i] = landmarks[i];
        }
    }
}