import cv2
from google.colab.patches import cv2_imshow
from fmd.mark_dataset.util import draw_marks
import numpy as np
from face_detector.detector import Detector
import tensorflow as tf

class predictor():
  def __init__(self, detector_face_model_path, cnn_model_path, threshold=0.7):
    self.detector_face_model = Detector(detector_face_model_path)
    self.cnn_model = tf.keras.models.load_model(cnn_model_path)
    self.threshold = threshold

  def predict(self, image_path):
    image = cv2.imread(image_path)
    _image = self.detector_face_model.preprocess(image)
    threshold = 0.7
    boxes, scores, _ = self.detector_face_model.predict(_image, threshold)

    # Transform the boxes into squares.
    boxes = self.detector_face_model.transform_to_square(
        boxes, scale=1.22, offset=(0, 0.13))

    # Clip the boxes if they cross the image boundaries.
    boxes, _ = self.detector_face_model.clip_boxes(
        boxes, (0, 0, image.shape[0], image.shape[1]))
    boxes = boxes.astype(np.int32)

    if boxes.size > 0:
      for facebox in boxes:
          # Crop the face image
          top, left, bottom, right = facebox
          top, left, bottom, right = int(top), int(left), int(bottom), int(right) 

          face_image = image[top:bottom, left:right]

          # Preprocess it.
          face_image = cv2.resize(face_image, (128, 128))
          face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

          cv2_imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

          out = self.cnn_model.predict(np.expand_dims(face_image, axis=0))[0]
          output_points = [[(out[i]),out[i+1]] for i in range(0,out.shape[0],2)]
          draw_marks(face_image,output_points)
          cv2_imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

