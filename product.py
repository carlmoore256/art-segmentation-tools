import svgwrite

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