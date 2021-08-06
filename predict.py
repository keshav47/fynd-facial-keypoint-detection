import cv2
from google.colab.patches import cv2_imshow
from fmd.mark_dataset.util import draw_marks
import numpy as np

def predict(image_path,detector_face,cnn_model):
  image = cv2.imread(image_path)
  _image = detector_face.preprocess(image)
  threshold = 0.7
  boxes, scores, _ = detector_face.predict(_image, threshold)

  # Transform the boxes into squares.
  boxes = detector_face.transform_to_square(
      boxes, scale=1.22, offset=(0, 0.13))

  # Clip the boxes if they cross the image boundaries.
  boxes, _ = detector_face.clip_boxes(
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

        out = cnn_model.predict(np.expand_dims(face_image, axis=0))[0]
        output_points = [[(out[i]),out[i+1]] for i in range(0,out.shape[0],2)]
        draw_marks(face_image,output_points)
        cv2_imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
