import sys
sys.path.append("/content/art-segmentation-tools/deepsvg")
from skimage.measure import find_contours
from svgpathtools import Path, Line, smoothed_path, paths2Drawing, svgstr2paths
from IPython.display import SVG, display
from deepsvg.svglib.svg_path import SVGPath
from deepsvg.svglib.svg import SVG as DeepSVG

def svg_path_from_lines(points):
    path = Path()
    for i in range(len(points)-1):
        path.append(Line(complex(points[i, 1], points[i, 0]), complex(points[i+1, 1], points[i+1, 0])))
    # close the path
    path.append(Line(complex(points[-1, 1], points[-1, 0]), complex(points[0, 1], points[0, 0])))
    return path

# returns a Path
def trace_image(img, smoothed=False):
  contours = find_contours(img)[0]
  path = svg_path_from_lines(contours)
  if smoothed:
    path = smoothed_path(path)
  return path

def show_svg(svg_doc):
    display(SVG(svg_doc))

# take a collection of Masks and trace them as an SVG
def masks_to_svg(masks) -> str:
  # trace the binary mask
  traced_paths = [trace_image(m.data) for m in masks]
  # convert to a drawing, then to string
  return paths2Drawing(traced_paths).tostring()


def simplify_svg(svg_str : str) -> DeepSVG:
  deep_svg = DeepSVG.from_str(svg_str)
  simplified = deep_svg.simplify_heuristic()
  return simplified