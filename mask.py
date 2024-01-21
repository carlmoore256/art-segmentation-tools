import numpy as np
from image import Image
from PIL import Image as PILImage, ImageDraw

class Mask():

  def __init__(self, mask_data):
    self.mask_data = mask_data

  @property
  def data(self):
    return self.mask_data
  
  @property
  def shape(self):
    return self.mask_data.shape
  
  @property
  def fill_ratio(self):
    return np.count_nonzero(self.mask_data) / (self.mask_data.shape[0] * self.mask_data.shape[1])

  def to_array(self, channels=3):
    arr = np.zeros((
        self.mask_data.shape[0], 
        self.mask_data.shape[1], 
        channels
    ))
    arr[self.mask_data] = 1
    return arr

  def to_image(self, channels=3, color=[1,1,1]):
    image_data = self.to_array(channels)
    image_data_orig = image_data.copy()
    if color is None:
      color = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
      image_data[:,:,i] = color[i]
    image_data *= image_data_orig
    return Image.from_data(image_data)

  def show(self, figsize=(4,4), title=None):
    if title is None:
      title = "Mask"
    Image.plot(self.mask_data, figsize, title)

  def copy(self):
    return Mask(self.mask_data.copy())

  def invert(self):
    return Mask(np.logical_not(self.mask_data))

  def plot_overlay(self):
    return self.to_image(4).plot()

  def apply(self, array):
    mask_temp = self.mask_data.copy()
    if len(array.shape) == len(self.mask_data.shape) + 1:
      mask_temp = np.expand_dims(mask_temp, -1)
    return array * mask_temp
  
  def get_writeable_data(self):
    mask_data = self.mask_data.copy()
    if np.max(mask_data) <= 1:
      mask_data = np.clip(mask_data*255, 0, 255)
    return mask_data.astype(np.uint8)
  
  def __iadd__(self, other):  
    if isinstance(other, Mask):
      self.mask_data += other.mask_data
    elif isinstance(other, (np.ndarray)):
      self.mask_data += other
    elif isinstance(other, (float, int)):
      self.mask_data += other
    else:
        return NotImplemented
    return self

  def __isub__(self, other):
    if isinstance(other, Mask):
      self.mask_data -= other.mask_data
    elif isinstance(other, (np.ndarray)):
      self.mask_data -= other
    elif isinstance(other, (float, int)):
      self.mask_data -= other
    else:
        return NotImplemented
    return self

  def __imul__(self, other):
    if isinstance(other, Mask):
      self.mask_data *= other.mask_data
    elif isinstance(other, (np.ndarray)):
      self.mask_data *= other
    elif isinstance(other, (float, int)):
      self.mask_data *= other
    else:
        return NotImplemented
    return self

  def __itruediv__(self, other):
    if isinstance(other, Mask):
      self.mask_data /= other.mask_data
    elif isinstance(other, (np.ndarray)):
      self.mask_data /= other
    elif isinstance(other, (float, int)):
      self.mask_data /= other
    else:
        return NotImplemented
    return self

  def __ipow__(self, other):
    if isinstance(other, Mask):
      self.mask_data **= other.mask_data
    elif isinstance(other, (np.ndarray)):
      self.mask_data **= other
    elif isinstance(other, (float, int)):
      self.mask_data **= other
    else:
        return NotImplemented
    return self

  def __eq__(self, other):
      if isinstance(other, Mask):
          return np.array_equal(self.mask_data, other.mask_data)
      else:
          return NotImplemented
  

class AnnotationMask(Mask):

  def __init__(self, data):
    super().__init__(data['segmentation'])
    # self.segmentation = data['segmentation']
    self.area = data['area']
    self.bbox = data['bbox']
    self.predicted_iou = data['predicted_iou']
    self.point_coords = data['point_coords']
    self.stability_score = data['stability_score']
    self.crop_box = data['crop_box']

    self.score = {
        "predicted_iou" : self.predicted_iou,
        "stability"     : self.stability_score
    }

  def get_score(self):
    return f"Predicted IOU: {round(self.predicted_iou, 4)} | Stability: {round(self.stability_score, 4)}"

  def to_annotation(self):
    return {
        "segmentation" : self.mask_data,
        "area"         : self.area,
        "bbox"         : self.bbox,
        "predicted_iou": self.predicted_iou,
        "point_coords" : self.point_coords,
        "stability_score": self.stability_score,
        "crop_box"     : self.crop_box
    }


  def __repr__(self) -> str:
    return f'Annotation Mask | area: {self.area}'
  

def create_mask_from_path(path, image_shape, scale_path=True):
    mask = PILImage.new('L', (image_shape[0], image_shape[1]), 0)
    draw = ImageDraw.Draw(mask)
    if scale_path:
      path = [(p[0] * image_shape[0], p[1] * image_shape[1]) for p in path]
    draw.polygon(path, fill=255)
    arr = np.array(mask).astype(np.float32)
    arr[arr == 255] = 1
    return Mask(arr)

    # return Mask(np.array(mask))
    # Create an empty mask
    # mask = np.zeros(image_shape, dtype=np.uint8)
    # # Fill the mask
    # cv2.fillPoly(mask, [np.array(path)], 255)
    # return mask