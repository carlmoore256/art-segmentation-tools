import argparse
from meshing import SVGMesher

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--outdir", type=str, required=True)
    args.add_argument("--depth_mult", type=float, required=False, default=100.0)
    args.add_argument("--sample_density", type=float, required=False, default=0.004)
    args.add_argument("--simplify_eps", type=float, required=False, default=0.005)
    args = args.parse_args()
    mesher = SVGMesher(args.input)
    print(f'Loaded SVG from {args.input} with {len(mesher.paths)} paths, tesselating polygons...')
    mesher.mesh({
        "sample_density": args.sample_density,
        "simplify_eps": args.simplify_eps
    })
    print(f'Generating and applying depth map...')
    mesher.apply_depth_map(args.depth_mult)
    print(f'Exporting to {args.outdir}')
    mesher.export(args.outdir, args.name)
    print(f'Finished!')
