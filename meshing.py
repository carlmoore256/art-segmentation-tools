from svgpathtools import Path, Line, smoothed_path, paths2Drawing, svgstr2paths, svg2paths2
import matplotlib.pyplot as plt
from path_operations import simplify_ramer_douglas_peucker, segments_to_path, line_norm_coords, plot_simplification, get_paths_extent, path_eccentricity, bbox_range
from segmented_image import get_image_from_svg
from triangulation import plot_triangles_triangulation, run_constrained_delaunay_safe
from mask import create_mask_from_path
import triangle
import numpy as np
from geometric_sampling import poisson_disk_sampling_within_path, centroidal_voronoi_tessellation_within_path, quad_points_within_path, generate_mesh_density_map, thin_points_based_on_density, generate_sobel_density_map
from depth_mapping import create_depth_map
from image_processing import get_image_gradients, plot_image_gradients, crop_depth_map, generate_normal_map, unletterbox
from tqdm import tqdm
from image import Image

import os
import json


class MeshedSVGPathResult:

    def __init__(self, path, segment_data, tesselation, parameters, depth_map=None):
        self.path = path
        self.segment_data = segment_data
        self.tesselation = tesselation
        self.parameters = parameters
        self.depth_map = depth_map
        self.bbox = path.bbox()
        self.depth_multiplier = 1.0
        self.global_depth = 0 # the depth that this path is relative to other paths 

    @property
    def sample_density(self):
        return self.parameters["sample_density"]

    @property
    def simplify_eps(self):
        return self.parameters["simplify_eps"]

    @property
    def min_sample_dist(self):
        return self.parameters["min_sample_dist"]

    @property
    def percent_reduction(self):
        return self.segment_data["percent_reduction"]

    @property
    def num_verts(self):
        return len(self.tesselation["triangulation"]["vertices"])

    @property
    def num_tris(self):
        return len(self.tesselation["triangulation"]["triangles"])

    @property
    def uvs(self):
        return self.tesselation["uvs"]

    @property
    def vertices(self):
        return self.tesselation["triangulation"]["vertices"]

    @property
    def triangles(self):
        return self.tesselation["triangulation"]["triangles"]
    
    @property
    def mask(self):
        return self.segment_data["image"]["mask"]
    
    @property
    def texture(self):
        mask = self.segment_data["image"]["mask"]
        image = self.segment_data["image"]["crop"]
        masked_image = image.new_from_mask(mask)
        return masked_image
    
    @property
    def image(self):
        return self.segment_data["image"]["crop"]

    def set_depth_map(self, depth_map, multiplier=1.0, global_depth=0):
        self.depth_map = depth_map
        self.depth_multiplier = multiplier
        self.global_depth = global_depth

    def plot_simplification(self):
        plot_simplification(self.segment_data["path"]["scaled"]["original"],
                            self.segment_data["path"]["scaled"]["simplified"])

    def plot_triangulation(self):
        plot_triangles_triangulation(self.tesselation["triangulation"])

    def generate_normal(self, ksize=3, scale=1, delta=0, norm_range=(0, 1)):
        normal_map = generate_normal_map((self.image.get_writeable_data()), ksize, scale, delta, norm_range)
        return normal_map

    def sample_depth(self, x, y, neighborhood=5):
        if self.depth_map is None:
            raise ValueError("Depth map not set")
        # Scale x and y to the depth map size
        x_scaled = min(int(x * self.depth_map.shape[1]), self.depth_map.shape[1] - 1)
        y_scaled = min(int(y * self.depth_map.shape[0]), self.depth_map.shape[0] - 1)
        # Define the neighborhood bounds
        half_neighborhood = neighborhood // 2
        x_min = max(x_scaled - half_neighborhood, 0)
        x_max = min(x_scaled + half_neighborhood + 1, self.depth_map.shape[1] - 1)
        y_min = max(y_scaled - half_neighborhood, 0)
        y_max = min(y_scaled + half_neighborhood + 1, self.depth_map.shape[0] - 1)

        # Extract the neighborhood and compute the average depth
        neighborhood_depth = self.depth_map[y_min:y_max, x_min:x_max]
        valid_depths = neighborhood_depth[neighborhood_depth > 0]  # Assuming 0 is invalid
        if valid_depths.size == 0:
            return 0  # Return a default value if no valid depths are found
        return np.mean(valid_depths)

    def export_obj(self, file_path):
        with open(file_path, "w") as file:
            for i, vert in enumerate(self.vertices):
                z = 0
                if self.depth_map is not None:
                    uv = self.uvs[i]
                    z = self.sample_depth(uv[0], 1-uv[1], 5) * self.depth_multiplier
                    # x = min(int(uv[0] * self.depth_map.shape[1]), self.depth_map.shape[1] - 1)
                    # y = min(int(uv[1] * self.depth_map.shape[0]), self.depth_map.shape[0] - 1) 
                    # z = self.depth_map[y, x, 0] * self.depth_multiplier
                file.write(f"v {vert[0]} {vert[1]} {z}\n")
            for uv in self.uvs:
                file.write(f"vt {uv[0]} {uv[1]}\n")
            # Write faces
            for tri in self.triangles:
                # OBJ face indices are 1-based and include vertex/texture-coordinate
                tri_indices = [f"{index + 1}/{index + 1}" for index in tri]
                file.write("f " + " ".join(tri_indices) + "\n")

    def get_metadata(self, relative_path=""):
        return {
            "meshFile": f"{relative_path}mesh.obj",
            "textureFile": f"{relative_path}texture.png",
            "normalFile": f"{relative_path}normal.png",
            "bbox": self.path.bbox(),
            "area": self.path.area(),
            "sampleDensity": self.sample_density,
            "simplifyEps": self.simplify_eps,
            "minSampleDist": self.min_sample_dist,
            "numVerts": self.num_verts,
            "globalDepth": float(self.global_depth),
        }

    def export_metadata(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.get_metadata(), file)

    def export_texture(self, file_path):
        self.texture.save(file_path)

    def export_normal(self, file_path):
        normal_map = self.generate_normal()
        normal_map = Image.from_data(normal_map)
        normal_map.save(file_path)

    def export(self, output_root, name, include_metadata=True):
        output_dir = os.path.join(output_root, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        obj_path = os.path.join(output_dir, "mesh.obj")
        self.export_obj(obj_path)
        texture_path = os.path.join(output_dir, "texture.png")
        self.export_texture(texture_path)
        normal_path = os.path.join(output_dir, "normal.png")
        self.export_normal(normal_path)

        if include_metadata:
            metadata_path = os.path.join(output_dir, "metadata.json")
            self.export_metadata(metadata_path, obj_path)


class SVGMesher:

    def __init__(self, svg_path):
        self.svg_path = svg_path
        self.image = get_image_from_svg(svg_path)
        self.paths, self.dpaths, self.svg_metadata = svg2paths2(svg_path)
        self.mesh_data = []
        self.depth_map = None
        self.bbox = get_paths_extent(self.paths)

    def mesh(self, parameters):
        self.mesh_data = mesh_all_svg_paths(
            self.svg_path,
            **parameters
        )

    def export(self, output_root, name):
        if len(self.mesh_data) == 0:
            raise Exception("No mesh data to export, run mesh() first")
        if not os.path.exists(output_root):
            raise Exception(f"Output root {output_root} does not exist")

        output_dir = os.path.join(output_root, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_layers = []
        for i, mesh_data in enumerate(self.mesh_data):
            layer_dir = f"layer_{i}"
            mesh_data.export(output_dir, layer_dir, include_metadata=False)
            all_layers.append(mesh_data.get_metadata(f"{layer_dir}/"))

        # save the image
        self.image.save(os.path.join(output_dir, f"full_image.png"))

        with open(os.path.join(output_dir, f"metadata.json"), "w") as file:
            json.dump({
                "name": name,
                "imageFile": f"{name}.png",
                "meshes": all_layers,
                **self.metadata
            }, file, indent=2)
        print(f"Finished exporting {len(self.mesh_data)} layers to {output_dir} for {name}")

    def apply_depth_map(self, 
                        depth_multiplier=10.0, 
                        depth_map=None, 
                        edge_removal=True, 
                        value_percentile_range=(8, 100), 
                        edge_params={
                            "threshold": 0.5,
                            "kern_size": 3,
                            "iterations": 3,
                            "blur": 21
                        }
    ):
        if depth_map is None:
            if self.depth_map is None:
                self._create_depth_map()
            depth_map = self.depth_map            
        for result in self.mesh_data:
            segment_depthmap, avg_segment_depth = crop_depth_map(
                depth_map, 
                result.path.bbox(), 
                result.segment_data["image"]["mask"],
                edge_removal,
                value_percentile_range[0],
                value_percentile_range[1],
                edge_params)
            result.set_depth_map(segment_depthmap, depth_multiplier, avg_segment_depth)
        print(f'Finished applying depth map to all layers')
    
    def show_depth_map(self):
        if self.depth_map is None:
            self._create_depth_map()
        plt.imshow(self.depth_map)
        plt.show()

    def _create_depth_map(self):
        print(f'Computing depth map')
        image_data = self.image.get_writeable_data()
        image_data, (diff_width, diff_height) = unletterbox(image_data)
        depth_map = create_depth_map(image_data)
        print(f'max of depth map is {np.max(depth_map)}')
        # pad the depth map back out to the original size
        depth_map_image = Image.from_data(depth_map)
        depth_map_image.pad_to_square()
        self.depth_map = depth_map_image.image_data


    @property
    def metadata(self):
        if len(self.mesh_data) == 0:
            return {}
        return {
            "bbox": self.bbox,
            "numPaths": len(self.paths),
            "numMeshes": len(self.mesh_data),
            "numVerts": sum([m.num_verts for m in self.mesh_data]),
            "area": self.image.width * self.image.height,
            "width": self.image.width,
            "height": self.image.height,
        }


def calculate_uv_coordinates(vertices, bbox):
    xmin, xmax, ymin, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    uv_coordinates = []
    for vertex in vertices:
        u = (vertex[0] - xmin) / width
        v = (vertex[1] - ymin) / height
        uv_coordinates.append((u, v))
    return uv_coordinates


def mask_and_simplify_path(svg_path, image, simplify_eps=0.001):
    bbox = svg_path.bbox()
    segments = [line_norm_coords(segment, bbox) for segment in svg_path]
    # flip y axis
    segments = [((p[0][0], 1-p[0][1]), (p[1][0], 1-p[1][1])) for p in segments]
    path = segments_to_path(segments)
    path_simple = simplify_ramer_douglas_peucker(path, simplify_eps)

    image_segment = image.get_cropped(bbox)

    norm_path_data = {
        "original": path,
        "simplified": path_simple,
    }

    xrange, yrange = bbox_range(bbox)
    scaled_path_data = {
        "original": [(p[0] * xrange, p[1] * yrange) for p in path],
        "simplified": [(p[0] * xrange, p[1] * yrange) for p in path_simple],
    }

    mask = create_mask_from_path(
        scaled_path_data["simplified"], (image_segment.shape[1], image_segment.shape[0]), False)
    # flip the mask vertically
    mask.mask_data = np.flip(mask.mask_data, 0)

    full_scaled_path_data = {
        "original": [((p[0] * xrange) + bbox[0], image.height - (bbox[3] - (p[1] * yrange))) for p in path],
        "simplified": [((p[0] * xrange) + bbox[0],  image.height - (bbox[3] - (p[1] * yrange))) for p in path_simple],
    }

    full_image_mask = create_mask_from_path(
        full_scaled_path_data["simplified"], (image.shape[1], image.shape[0]), False)
    # flip the mask vertically
    full_image_mask.mask_data = np.flip(full_image_mask.mask_data, 0)

    return {
        "path": {
            "normalized": norm_path_data,
            "scaled": scaled_path_data,
            "full_scaled": full_scaled_path_data
        },
        "image": {
            "mask": mask,
            "crop": image_segment,
            "full_mask": full_image_mask
        },
        "percent_reduction": 1 - (len(path_simple) / len(path))
    }


def tesselate_polygon(
        polygon,
        bbox,
        sampling_fn=lambda x: poisson_disk_sampling_within_path(x, 0.05),
        mesh_density_map=None
):
    internal_points = sampling_fn(polygon)
    if len(internal_points) == 0:
        internal_points = None
    if mesh_density_map is not None:
        internal_points = thin_points_based_on_density(
            internal_points, mesh_density_map, 3, 2)

    triangulation = run_constrained_delaunay_safe(
        polygon,
        internal_points)

    xrange, yrange = bbox_range(bbox)
    # expand out range
    triangulation['vertices'] = [(p[0] * xrange, p[1] * yrange)
                                 for p in triangulation['vertices']]
    uvs = calculate_uv_coordinates(
        triangulation['vertices'], [0, xrange, 0, yrange])
    return {
        "triangulation": triangulation,
        "uvs": uvs
    }


def mesh_svg_path(
        path,
        image,
        sample_density=0.3,
        simplify_eps=0.005,  # higher value = greater simplification
        # if True, segments will be simplified proportionally to their size
        dynamic_simplify=True,
        display=False,
        verbose=False) -> MeshedSVGPathResult:

    bbox = path.bbox()
    area = path.area()
    image_area = image.width * image.height
    percent_area = area / image_area

    if dynamic_simplify:
        # if the path is large, we can simplify more
        simplify_eps = simplify_eps * (1-percent_area)

    segment_data = mask_and_simplify_path(path, image, simplify_eps)

    # min_sample_dist = (((1/sample_density) * 100) * (1/percent_area)) / image_area
    # Adding a small constant to avoid division by zero
    scaling_factor = 1.0 / (percent_area + 0.01)
    min_sample_dist = sample_density * scaling_factor
    tesselation = tesselate_polygon(
        segment_data["path"]["normalized"]["simplified"],
        bbox,
        lambda x: poisson_disk_sampling_within_path(x, min_sample_dist)
        # mesh_density_map
    )

    if verbose:
        print(f"Polygon area {(percent_area * 100):.2f}% | Simplified Reduction {segment_data['percent_reduction']} | Sample Dist {min_sample_dist:.3f} | Num Verts {len(tesselation['triangulation']['vertices'])} | Num Tris {len(tesselation['triangulation']['triangles'])}", end="\r")

    if display:
        plot_simplification(segment_data["path"]["scaled"]["original"],
                            segment_data["path"]["scaled"]["simplified"])
        plot_triangles_triangulation(tesselation["triangulation"])

    result = MeshedSVGPathResult(path, segment_data, tesselation, {
        "sample_density": sample_density,
        "simplify_eps": simplify_eps,
        "min_sample_dist": min_sample_dist,
    })

    return result


def mesh_all_svg_paths(
        svg_path,
        sample_density=0.3,
        simplify_eps=0.005,
        dynamic_simplify=False,
        min_area=0.0001,
        max_eccentricity=75,
    ):
    print(f'Meshing with sample density {sample_density} and simplify eps {simplify_eps}')
    paths, dpaths, metadata = svg2paths2(svg_path)
    image = get_image_from_svg(svg_path)
    # sort paths by area
    paths = sorted(paths, key=lambda p: p.area(), reverse=True)
    orig_len = len(paths)
    # filter out paths that are really tall or wide
    paths = list(filter(lambda p: path_eccentricity(p) < max_eccentricity, paths))
    paths = list(filter(lambda p: p.area() > 0, paths))
    if len(paths) == 0:
        raise Exception("No paths remain after filtering")
    max_area = paths[0].area()
    paths = list(filter(lambda p: p.area() / max_area > min_area, paths))

    print(f'Filtered out {orig_len - len(paths)} paths with area < {min_area} or eccentricity > {max_eccentricity} | Remaining paths: {len(paths)}')

    total_verts = 0
    all_segment_data = []
    for i, p in enumerate(paths):
        print(f'\nProcessing path {i+1}/{len(paths)} | Total verts: {total_verts}', end="\r")
        result = mesh_svg_path(
            p, image, sample_density, simplify_eps, dynamic_simplify, False, False)
        all_segment_data.append(result)
        total_verts += result.num_verts
    return all_segment_data
