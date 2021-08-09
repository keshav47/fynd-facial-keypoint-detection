import os
import time
import cv2
import numpy as np
import tensorflow as tf

import fmd
from fmd.mark_dataset.dataset import MarkDataset
from dataset import get_parsed_dataset

def representative_dataset_gen():
    wflw_dir = "./wflw_data"
    ds_wflw = fmd.wflw.WFLW(False, "wflw_test")
    ds_wflw.populate_dataset(wflw_dir)

    for _ in range(100):
        sample = ds_wflw.pick_one()

        # Get image and marks.
        image = sample.read_image()
        marks = sample.marks

        # Crop the face out of the image.
        image_cropped, _, _ = crop_face(image, marks, scale=1.2)

        # Get the prediction from the model.
        image_cropped = cv2.resize(image_cropped, (128, 128))
        img_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        img_input = normalize(np.array(img_rgb, dtype=np.float32))

        yield [np.expand_dims(img_input, axis=0)]

def quantize(saved_model, mode=None, representative_dataset=None):
    """TensorFlow model quantization by TFLite.
    Args:
        saved_model: the model's directory.
        mode: the quantization mode.
    Returns:
        a tflite model quantized.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model("./exported")

    # By default, do Dynamic Range Quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Integer With Float Fallback
    if mode["IntegerWithFloatFallback"]:
        converter.representative_dataset = representative_dataset

    # Integer only.
    if mode["IntergerOnly"]:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8

    # Float16 only.
    if mode["FP16"]:
        converter.target_spec.supported_types = [tf.float16]

    # [experimental] 16-bit activations with 8-bit weights
    if mode["16x8"]:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

    # Finally, convert the model.
    tflite_model = converter.convert()

    return tflite_model

def crop_face(image, marks, scale=1.8, shift_ratios=(0, 0)):
    """Crop the face area from the input image.

    Args:
        image: input image.
        marks: the facial marks of the face to be cropped.
        scale: how much to scale the face box.
        shift_ratios: shift the face box to (right, down) by facebox size * ratios

    Returns:
        cropped face image.
    """
    # How large the bounding box is?
    x_min, y_min, _ = np.amin(marks, 0)
    x_max, y_max, _ = np.amax(marks, 0)
    side_length = max((x_max - x_min, y_max - y_min)) * scale

    # Face box is scaled, get the new corners, shifted.
    img_height, img_width, _ = image.shape
    x_shift, y_shift = np.array(shift_ratios) * side_length

    x_start = int(img_width / 2 - side_length / 2 + x_shift)
    y_start = int(img_height / 2 - side_length / 2 + y_shift)
    x_end = int(img_width / 2 + side_length / 2 + x_shift)
    y_end = int(img_height / 2 + side_length / 2 + y_shift)

    # In case the new bbox is out of image bounding.
    border_width = 0
    border_x = min(x_start, y_start)
    border_y = max(x_end - img_width, y_end - img_height)
    if border_x < 0 or border_y > 0:
        border_width = max(abs(border_x), abs(border_y))
        x_start += border_width
        y_start += border_width
        x_end += border_width
        y_end += border_width
        image_with_border = cv2.copyMakeBorder(image, border_width,
                                               border_width,
                                               border_width,
                                               border_width,
                                               cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
        image_cropped = image_with_border[y_start:y_end,
                                          x_start:x_end]
    else:
        image_cropped = image[y_start:y_end, x_start:x_end]

    return image_cropped, border_width, (x_start, y_start, x_end, y_end)


def normalize(inputs):
    """Preprocess the inputs. This function follows the official implementation
    of HRNet.
    Args:
        inputs: a TensorFlow tensor of image.
    Returns:
        a normalized image.
    """
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalization
    return ((inputs / 255.0) - img_mean)/img_std

def inference(tflite_model):
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_index,output_index = interpreter.get_input_details()[0]["index"],interpreter.get_output_details()[0]["index"]
  result_mse = []
  start = time.time()
  for input_image,output_val in dataset_val:
    interpreter.set_tensor(input_index, input_image)
    output = interpreter.tensor(output_index)
    result_mse.append(mse(np.array(output_val), output()).numpy())

  return sum(result_mse)/len(result_mse),time.time()-start


if __name__ == "__main__":
    MODE = {"DynamicRangeQuantization": None,
        "IntegerWithFloatFallback": None,
        "IntergerOnly": None,
        "FP16": None,
        "16x8": None}
    # Construct a dataset for evaluation.
    dataset_val = get_parsed_dataset(record_file="/content/fynd-facial-keypoint-detection/tfrecord/test_wflw.record",
                                        batch_size=1,
                                        shuffle=False)
    mse = tf.keras.losses.MeanSquaredError()
    # The directory to save quantized models.
    export_dir = "./tflite_optimized"

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # The model to be quantized.
    saved_model = "./exported"
    # Dynamic range quantization
    mode = MODE.copy()
    mode.update({"DynamicRangeQuantization": True})
    tflite_model = quantize(saved_model, mode)
    print(f"Mean Square Error for DynamicRangeQuantization is {inference(tflite_model)[0]}, speed of inference {inference(tflite_model)[1]}s")

    
    mode = MODE.copy()
    mode.update({"IntegerWithFloatFallback": True})
    tflite_model = quantize(saved_model, mode, representative_dataset_gen)
    open("./tflite_optimized/quant_int_fp_fallback.tflite", "wb").write(tflite_model)
    print(f"Mean Square Error for IntegerWithFloatFallback is {inference(tflite_model)[0]}, speed of inference {inference(tflite_model)[1]}s")


    mode = MODE.copy()
    mode.update({"IntegerOnly": True})
    tflite_model = quantize(saved_model, mode,  representative_dataset_gen)
    open("./tflite_optimized/quant_int_only.tflite", "wb").write(tflite_model)
    print(f"Mean Square Error for IntegerOnly is {inference(tflite_model)[0]}, speed of inference {inference(tflite_model)[1]}s")
    

    # Float16 quantization
    mode = MODE.copy()
    mode.update({"FP16": True})
    tflite_model = quantize(saved_model, mode)
    open("./tflite_optimized/quant_fp16.tflite", "wb").write(tflite_model)
    print(f"Mean Square Error for FP16 is {inference(tflite_model)[0]}, speed of inference {inference(tflite_model)[1]}s")
    

    # 16x8 quantization
    mode = MODE.copy()
    mode.update({"16x8": True})
    tflite_model = quantize(saved_model, mode)
    open("./tflite_optimized/quant_16x8.tflite", "wb").write(tflite_model)
    print(f"Mean Square Error for 16x8 is {inference(tflite_model)[0]}, speed of inference {inference(tflite_model)[1]}s")
    