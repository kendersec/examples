package org.tensorflow.lite.examples.classification.onnx;

import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.examples.classification.Recognition;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class OnnxTensorUtil {

    public static final int IMAGE_HEIGHT = 224;
    public static final int IMAGE_WIDTH = 224;
    private static final float[] MEAN_ARRAY = new float[] {0.485f, 0.456f, 0.406f};
    private static final float[] STDDEV_ARRAY = new float[] {0.229f, 0.224f, 0.225f};

    private static final int MAX_RESULTS = 3;

    private OnnxTensorUtil() {}

    /*
# convert the input data into the float32 input
img_data = input_data.astype('float32')

#normalize
mean_vec = np.array([0.485, 0.456, 0.406])
stddev_vec = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype('float32')
for i in range(img_data.shape[0]):
    norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

#add batch channel
norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
return norm_img_data
     */

    public static OnnxTensor makeImageTensor(OrtEnvironment env, Bitmap bitmap, int orientation) throws OrtException {
        float[][][][] imageData = new float[1][3][IMAGE_WIDTH][IMAGE_HEIGHT];
        zeroData(imageData);

        Bitmap transformedImage = convertImage(bitmap, orientation);
        int[] intValues = new int[IMAGE_WIDTH * IMAGE_HEIGHT];
        transformedImage.getPixels(intValues, 0, IMAGE_WIDTH, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

        int i = 0;
        for(int w=0; w < IMAGE_WIDTH; ++w) {
            for(int h=0; h < IMAGE_HEIGHT; ++h) {
                imageData[0][0][w][h] = (((intValues[i] >> 16 & 255) / 255f) - MEAN_ARRAY[0]) / STDDEV_ARRAY[0];
                imageData[0][1][w][h] = (((intValues[i] >> 8 & 255) / 255f) - MEAN_ARRAY[1]) / STDDEV_ARRAY[1];
                imageData[0][2][w][h] = (((intValues[i] & 255) / 255f) - MEAN_ARRAY[2]) / STDDEV_ARRAY[2];
                i++;
            }
        }

        return OnnxTensor.createTensor(env, imageData);
    }

    /** Loads input image, and applies preprocessing */
    private static Bitmap convertImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRoration = sensorOrientation / 90;
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(IMAGE_HEIGHT, IMAGE_WIDTH, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(numRoration))
                        //.add(new NormalizeOp(0f, 255f))
                        .build();
        return imageProcessor.process(tensorImage).getBitmap();
    }

    private static void zeroData(float[][][][] data) {
        // Zero the array
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                for (int k = 0; k < data[i][j].length; k++) {
                    Arrays.fill(data[i][j][k], 0.0f);
                }
            }
        }
    }

    public static float[] resultToArray(OrtSession.Result result) throws OrtException {
        float[] res1d = ((float[][]) result.get(0).getValue())[0];
        return softmax(res1d);
    }

    public static List<Recognition> getTopKProbability(float[] results, List<String> labels) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i=0; i < results.length; ++i) {
            pq.add(new Recognition(labels.get(i), labels.get(i), results[i], null));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    public static float[] softmax(float[] input) {
        double[] tmp = new double[input.length];
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            double val = Math.exp(input[i]);
            sum += val;
            tmp[i] = val;
        }

        float[] output = new float[input.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = (float) (tmp[i] / sum);
        }

        return output;
    }
}
