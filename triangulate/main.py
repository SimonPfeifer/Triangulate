from cv2 import imread, imwrite

from triangulate import triangulate


def main(args):
    corner = args.corner
    if corner == None:
        corner = 5
    
    radial = args.radial
    if radial == None:
        radial = 25

    img = imread(args.input)
    img_new = triangulate(img, corner_threshold=corner, radial_min_sep=radial,
                          profile=args.verbose)
    imwrite(args.output, img_new)

if __name__ == '__main__':
    try:
        import argparse
    except ImportError:
        print('Make sure you are using python 2.7+')
        raise

    parser = argparse.ArgumentParser(description='Triangulate')
    parser.add_argument('-i', '--input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output image file path')
    parser.add_argument('--corner', type=int, help='FAST corner detection threshold')
    parser.add_argument('--radial', type=int, help='Minimum radial separation of nodes')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
