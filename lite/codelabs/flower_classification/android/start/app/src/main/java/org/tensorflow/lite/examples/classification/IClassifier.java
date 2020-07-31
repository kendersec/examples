package org.tensorflow.lite.examples.classification;

import android.graphics.Bitmap;

import java.util.List;

public interface IClassifier {
    List<Recognition> recognizeImage(Bitmap bitmap, int sensorOrientation);

    void close();

    int getImageSizeX();

    int getImageSizeY();
}
