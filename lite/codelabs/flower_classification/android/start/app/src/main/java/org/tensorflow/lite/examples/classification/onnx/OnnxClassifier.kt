package org.tensorflow.lite.examples.classification.onnx

import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.examples.classification.IClassifier
import org.tensorflow.lite.examples.classification.Recognition
import org.tensorflow.lite.support.common.FileUtil
import java.util.*


class OnnxClassifier(val context: Context): IClassifier {

    private val env: OrtEnvironment by lazy {
        OrtEnvironment.getEnvironment()
    }

    private val session: OrtSession by lazy {
        val modelBytes = context.assets.open("mobilenetv2-1.0.onnx").readBytes()
        val session = env.createSession(modelBytes)

        Log.d("davmart", "Number of inputs: ${session.numInputs}")
        Log.d("davmart", "Number of outputs: ${session.numOutputs}")
        Log.d("davmart", "Input names: ${session.inputNames}")
        Log.d("davmart", "Output names: ${session.outputNames}")
        Log.d("davmart", "Input info: ${session.inputInfo}")
        Log.d("davmart", "Output info: ${session.outputInfo}")

        session
    }

    // Loads labels out from the label file.
    private val labels: List<String> by lazy {
        FileUtil.loadLabels(context, "labels.txt")
    }

    override fun recognizeImage(bitmap: Bitmap?, sensorOrientation: Int): List<Recognition> {
        val recognitions: List<Recognition> = listOf()

        bitmap?.let {
            val inputName = session.inputNames.iterator().next()
            val result = session.run(
                    Collections.singletonMap(
                            inputName,
                            OnnxTensorUtil.makeImageTensor(env, bitmap, sensorOrientation)))

            val categoryArray = OnnxTensorUtil.resultToArray(result)
            return OnnxTensorUtil.getTopKProbability(categoryArray, labels)
        }

        return recognitions
    }

    override fun getImageSizeY(): Int {
        return OnnxTensorUtil.IMAGE_HEIGHT;
    }

    override fun getImageSizeX(): Int {
        return OnnxTensorUtil.IMAGE_WIDTH;
    }

    override fun close() {
        session.close();
    }
}