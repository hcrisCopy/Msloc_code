"""Only prepare fixed small-data annotations; does not extract/train/evaluate."""
import argparse

from subset import prepare_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json",required=True)
    parser.add_argument("--test-json",required=True)
    parser.add_argument("--val-json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--max-videos",type=int,default=10)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()
    paths = dict(train=args.train_json,test=args.test_json)
    if args.val_json:
        paths["val"] = args.val_json
    prepare_subset(paths,args.output,args.max_videos,args.seed)


if __name__ == "__main__":
    main()
