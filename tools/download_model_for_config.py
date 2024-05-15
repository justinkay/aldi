import argparse
import os
from urllib.request import urlretrieve

from aldi.config import add_aldi_config


PTH_URL = 'https://github.com/justinkay/aldi/releases/download/v0.0.1/'

def get_cfg():
    from detectron2.config.defaults import _C
    return _C.clone()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def main(args):
    cfg = get_cfg()
    add_aldi_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    weights = cfg.MODEL.WEIGHTS
    if not os.path.exists(weights):
        dirname, filename = os.path.split(weights)
        os.makedirs(dirname, exist_ok=True)
        url = PTH_URL + filename
        print("Downloading", url, "to", weights, "...")
        urlretrieve(url, weights)

if __name__ == "__main__":
    args = parser().parse_args()
    main(args)