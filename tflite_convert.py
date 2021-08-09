import os
import tensorflow as tf

def tflite_convert(saved_model):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model


if __name__ == "__main__":
    # The directory to save quantized models.
    export_dir = "./tflite_optimized"

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # The model to be quantized.
    saved_model = "./exported"
    tflite_model = tflite_convert(saved_model)
    
    open("./optimized/keypoint.tflite", "wb").write(tflite_model)




