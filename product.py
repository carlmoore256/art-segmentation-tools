import svgwrite
# import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from svg_tools import masks_to_svg, simplify_svg
import os

def create_layered_svg(seg_image, outpath):
  width = seg_image.image.width
  height = seg_image.image.height
  doc = svgwrite.Drawing(size=(width, height))

  data_uri = seg_image.get_background().to_data_uri()

  element = svgwrite.container.Group(opacity=1)
  element.add(svgwrite.image.Image(href=data_uri, size=(width, height)))
  element['id'] = 'layer-0'
  doc.add(element)

  # sort the layers from largest to smallest
  sorted_masks = seg_image.get_masks_by_area()
  
  for i in range(len(sorted_masks)):
    layer = seg_image.image.new_from_mask(sorted_masks[i], with_alpha=True)
    # layer = seg_image.get_masked(i)
    data_uri = layer.to_data_uri()
    element = svgwrite.container.Group(opacity=1)
    element.add(svgwrite.image.Image(href=data_uri, size=(width, height)))
    element['id'] = f'layer-{i+1}'
    doc.add(element)

  doc.saveas(outpath)


# creates the svg bundle containing the original image, background image, and 
# segmentation paths
def create_svg_bundle(seg_image, outpath):
  width = seg_image.image.width
  height = seg_image.image.height
  doc = svgwrite.Drawing(size=(width, height))

  data_uri_bg = seg_image.get_background().to_data_uri()
  data_uri_orig = seg_image.image.to_data_uri()

  # add the original image
  element = svgwrite.container.Group(opacity=1)
  element.add(svgwrite.image.Image(href=data_uri_orig, size=(width, height)))
  element['id'] = 'image-orig'
  doc.add(element)

  # add the background image
  element = svgwrite.container.Group(opacity=1)
  element.add(svgwrite.image.Image(href=data_uri_bg, size=(width, height)))
  element['id'] = 'image-bg'
  doc.add(element)

  # sort the layers from largest to smallest
  sorted_masks = seg_image.get_masks_by_area()

  print(f'Tracing SVG with {len(sorted_masks)} masks')
  traced_svg = masks_to_svg(sorted_masks)
  print(f'Simplifying SVG')
  deep_svg = simplify_svg(traced_svg)

  # I know this is exhausting
  svg_str = deep_svg.to_str()
  soup = BeautifulSoup(svg_str, "xml")

  # iterate through all the paths, create new elements
  for i, path in enumerate(soup.find_all("path")):
    path_str = str(path)
    path = svgwrite.path.Path(path['d'])
    element = svgwrite.container.Group(opacity=1)
    element.add(path)
    element['id'] = f'layer-{i}'
    doc.add(element)
  doc.saveas(outpath)


# creates the svg bundle within a destination directory
def export_bundle(seg_image, artwork_name, root_dir, overwrite=False):
  filepath =  f"{root_dir}/{artwork_name}.svg"
  if os.path.exists(filepath) and not overwrite:
    raise Exception(f"File {filepath} already exists, choose another id, or set overwrite = True")
  create_svg_bundle(seg_image, filepath)
