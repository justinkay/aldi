import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.dirname(current_dir)
if par_dir not in sys.path:
    sys.path.append(par_dir)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
import matplotlib.pyplot as plt

from config import add_da_config
from trainer import DATrainer
import datasets # register datasets with Detectron2
import rcnn # register DA R-CNN model with Detectron2
import gaussian_rcnn # register Gaussian R-CNN model with Detectron2
import backbone # register Swin-B FPN backbone with Detectron2


def setup(args):
    """
    Copied directly from detectron2/tools/train_net.py
    """
    cfg = get_cfg()
    add_da_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class TestTrainer(DATrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def train(self):
        self.max_iter = 1
        super().train()

    def after_train(self):
        return

def main(args):

    import trainer
    trainer.DEBUG = True
    debug_dict = trainer.debug_dict

    cfg = setup(args)

    trainer = TestTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    if not args.no_save:

        labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong = debug_dict['last_labeled_weak'], debug_dict['last_labeled_strong'], debug_dict['last_unlabeled_weak'], debug_dict['last_unlabeled_strong']
        pseudolabeled = debug_dict['last_pseudolabeled']

        for i, (lw, ls, uw, pl) in enumerate(zip(labeled_weak, labeled_strong, unlabeled_weak, pseudolabeled)):
            fig, ax = plt.subplots(4,1, figsize=(20,20))
            labeled_im = lw['image'].permute(1,2,0).cpu().numpy()
            unlabeled_im = ls['image'].permute(1,2,0).cpu().numpy()
            unlabeled_before_im = uw['image'].permute(1,2,0).cpu().numpy()
            unlabeled_after_im = pl['image'].permute(1,2,0).cpu().numpy()

            # plot images
            ax[0].imshow(labeled_im[:,:,::-1])
            ax[1].imshow(unlabeled_im[:,:,::-1])
            ax[2].imshow(unlabeled_before_im[:,:,::-1])
            ax[3].imshow(unlabeled_after_im[:,:,::-1])

            # plot instances as rectangles
            for inst in lw['instances'].gt_boxes.tensor:
                x1,y1,x2,y2 = inst.cpu().numpy()
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                ax[0].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))
            for inst in ls['instances'].gt_boxes.tensor:
                x1,y1,x2,y2 = inst.cpu().numpy()
                ax[1].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))
            for inst in uw['instances'].gt_boxes.tensor:
                x1,y1,x2,y2 = inst.cpu().numpy()
                ax[2].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))
            for inst in pl['instances'].gt_boxes.tensor:
                x1,y1,x2,y2 = inst.cpu().numpy()
                ax[3].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=3))

            ax[0].set_title('Labeled Weak')
            ax[1].set_title('Labeled Strong')
            ax[2].set_title('Unlabeled Weak')
            ax[3].set_title('Unlabeled Strong (w/ Pseudo boxes)')

            plt.savefig(f'debug_{lw["image_id"]}_{i}.png')
            plt.close()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--no-save", action="store_true", help="don't save debug images")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
