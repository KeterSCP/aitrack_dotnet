using Emgu.CV;
using Emgu.CV.Structure;

namespace AITrackDotnet;

internal static class ImageProcessing
{
    private static readonly MCvScalar MeanScaling = new (0.485f, 0.456f, 0.406f);
    private static readonly MCvScalar StdScaling = new (0.229f, 0.224f, 0.225f);

    static ImageProcessing()
    {
        MeanScaling = new MCvScalar(MeanScaling.V0 / StdScaling.V0, MeanScaling.V1 / StdScaling.V1, MeanScaling.V2 / StdScaling.V2);
        StdScaling = new MCvScalar(StdScaling.V0 * 255f, StdScaling.V1 * 255f, StdScaling.V2 * 255f);
    }

    public static unsafe void NormalizeAndTranspose(Mat img, float[] dest, int x = 224, int y = 224)
    {
        var stride = x * y;

        var data = (float*)img.DataPointer;
        for (int channel = 0; channel < 3; channel++)
        {
            float stdScalingForChannel = channel switch
            {
                0 => (float)StdScaling.V0,
                1 => (float)StdScaling.V1,
                2 => (float)StdScaling.V2,
                _ => 0
            };

            float meanScalingForChannel = channel switch
            {
                0 => (float)MeanScaling.V0,
                1 => (float)MeanScaling.V1,
                2 => (float)MeanScaling.V2,
                _ => 0
            };

            for (int i = 0; i < stride; i++)
            {
                ref float fromRef = ref data[i * 3 + channel];

                fromRef /= stdScalingForChannel;
                fromRef -= meanScalingForChannel;

                dest[channel * stride + i] = fromRef;
            }
        }
    }
}