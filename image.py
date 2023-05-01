import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image as PImage

class Image():

  def __init__(self, filepath=None):
    self.image_data = None
    if filepath is not None:
      self.image_data = Image.load(filepath)

  @property
  def shape(self):
    return self.image_data.shape

  @property
  def width(self):
    return self.image_data.shape[0]

  @property
  def height(self):
    return self.image_data.shape[1]
    
  @property
  def has_alpha(self):
    if self.image_data.shape[-1] < 4:
      return False
    if self.image_data.shape[-1] == 4:
      return True
    else:
      raise Exception(f"[!] Error: Shape of image_data invalid: {self.image_data.shape}")

  @property
  def data(self):
    return self.image_data

  @staticmethod
  def load(filepath):
    image_data = cv2.imread(filepath)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB).astype(float)
    image_data /= 256
    return image_data

  @staticmethod
  def plot(image_data, title="", figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_data)
    ax.set_title(title)
    ax.axis('off')
    return fig

  @staticmethod
  def from_file(filepath):
    return Image(filepath)

  @staticmethod
  def from_data(image_data):
    image = Image()
    image.image_data = image_data
    return image

  @staticmethod
  def empty_image_like(image):
    image_data = np.zeros(image.shape)
    return Image.from_data(image_data)

  @property
  def alpha(self):
    return self.image_data[:,:,3]

  def copy(self):
    return Image.from_data(self.data.copy())

  def get_writeable_data(self):
    image_data = self.image_data.copy()
    if np.max(image_data) <= 1:
      image_data = np.clip(image_data*255, 0, 255)
    return image_data.astype(np.uint8)

  def save(self, outpath):
    color = cv2.COLOR_RGB2BGR
    if self.has_alpha:
      color = cv2.COLOR_RGBA2BGRA
    image_data = self.get_writeable_data()
    cv2.imwrite(outpath, cv2.cvtColor(image_data, color))    
    print(f'Saved image to {outpath}')

  def show(self, title="", figsize=(10,10)):
    Image.plot(self.image_data, title, figsize).show()

  def resize(self, dims, interpolation=None):
    if interpolation is None: # auto select interpolation
      # https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
      if dims[0]*dims[1] < self.image_data.shape[0] * self.image_data.shape[1]:
        interpolation = cv2.INTER_AREA
      else:
        interpolation = cv2.INTER_CUBIC
    self.image_data = cv2.resize(self.image_data, dims, interpolation=interpolation)

  def new_from_mask(self, mask, with_alpha=False):
    image = Image.from_data(mask.apply(self.image_data.copy()))
    if with_alpha:
      image.add_alpha(mask)
    return image

  def add_alpha(self, alpha=None, mult=1):
    if self.has_alpha:
      if alpha is None:
        print(f'[/!\] Warning: Image already has alpha')
        return
      else:
        self.image_data[:, :, 3] = alpha
        return
    alpha = np.ones((self.image_data.shape[0], self.image_data.shape[1], 1), dtype=np.uint8)    
    self.image_data = np.concatenate([self.image_data, alpha], axis=-1)

  def __iadd__(self, other):
    
    if isinstance(other, Image):
      self.image_data += other.image_data
    elif isinstance(other, (np.ndarray)):
      self.image_data += other
    elif isinstance(other, (float, int)):
      self.image_data += other
    else:
        return NotImplemented
    return self

  def __isub__(self, other):
    if isinstance(other, Image):
      self.image_data -= other.image_data
    elif isinstance(other, (np.ndarray)):
      self.image_data -= other
    elif isinstance(other, (float, int)):
      self.image_data -= other
    else:
        return NotImplemented
    return self

  def __imul__(self, other):
    if isinstance(other, Image):
      self.image_data *= other.image_data
    elif isinstance(other, (np.ndarray)):
      self.image_data *= other
    elif isinstance(other, (float, int)):
      self.image_data *= other
    else:
        return NotImplemented
    return self

  def __itruediv__(self, other):
    if isinstance(other, Image):
      self.image_data /= other.image_data
    elif isinstance(other, (np.ndarray)):
      self.image_data /= other
    elif isinstance(other, (float, int)):
      self.image_data /= other
    else:
        return NotImplemented
    return self

  def __ipow__(self, other):
    if isinstance(other, Image):
      self.image_data **= other.image_data
    elif isinstance(other, (np.ndarray)):
      self.image_data **= other
    elif isinstance(other, (float, int)):
      self.image_data **= other
    else:
        return NotImplemented
    return self

  def __eq__(self, other):
      if isinstance(other, Image):
          return np.array_equal(self.image_data, other.image_data)
      else:
          return NotImplemented


def alpha_blend_images(image_bottom, image_top, top_transparency=0.75):
  background = PImage.fromarray(image_bottom.get_writeable_data())
  foreground = PImage.fromarray((image_top.get_writeable_data() * top_transparency).astype(np.uint8))
  background.paste(foreground, (0,0), foreground)
  return Image.from_data(np.asarray(background))