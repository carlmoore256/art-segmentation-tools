# generate a normal for an input image
import argparse
from image_processing import generate_normal_map
import cv2
import os


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image", type=str, required=True)
    args.add_argument("--output", type=str, required=False)
    args.add_argument("--ksize", type=int, default=3)
    args.add_argument("--scale", type=int, default=1)
    args.add_argument("--delta", type=int, default=0)
    args.add_argument("--norm_range", type=tuple, default=(0, 1))

    args = args.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.image), "normal_" + os.path.basename(args.image)
        )
    image = cv2.imread(args.image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normal = generate_normal_map(image, ksize=args.ksize, scale=args.scale, delta=args.delta, norm_range=args.norm_range)
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output, normal)
