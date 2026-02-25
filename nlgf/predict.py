import argparse
import sys

from model import predict

def get_args():
    parser = argparse.ArgumentParser(
        description="Predict using the NLGF classifier.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30)
    )
    parser.add_argument("--article_link", help="Local news aricle's link")
    parser.add_argument("--publisher_longitude", help="Publisher's location's longitude")
    parser.add_argument("--publisher_latitude", help="Publisher's location's latitude")
    return parser


def process_args(args):

    if not args.article_link and not args.publisher_longitude and not args.publisher_latitude:
        raise ValueError("Prediction requires article_link, publisher_longitude, publisher_latitude")
    geo_focus_level, geo_focus = predict(args.article_link, args.publisher_longitude, args.publisher_latitude)
    print (f"Geo focus level: {geo_focus_level}")
    print (f"Geo focus: {geo_focus}")


def main():
    parser = get_args()
    args = parser.parse_args()
    process_args(args)


if __name__ == '__main__':
    main()