# DAOD MSc

## File structure 
We put the original aldi-project in its own directory, so that our own work sits outside with the authors' original work in the aldi-folder with minimal changes from us. 

`Common-configs`  contains the following types of models, which are generally reused in the experiments. The baseline and oracle 
backbone =  basic backbone model, like ResNet50 w. RPN - pre-trained on COCO or 
            imagenet or whatever
baseline =  the backbone w. transfer learning on SOURCE data - this is both the source-only 
            model AND the burn-in model (initialisation of student/teacher for the DAOD-method)
oracle =    the backebone w. transfer learning on TARGET data

```json
common-conifgs/ // config files for models that are common to different experiments (backbones and then baselines/oracles for different datasets as specified byt the folder name)
    backbones/ // backbones of different architectures (specified in filename) - the basic model used as backbone, before any transfer learning/domain adaptation
        detectron2/ // built in dectrecton configs for the RCNN-FPN-backbone
        Base-RCNN-FPN.yaml // ResNet50 w. FPN pretrained on COCO
    cityscapes/ // source- and oracle-only model configs for the cityscapes->foggycityscapes dataset
        ...
    ...
experiments // configs and output files for experiments we performed/models we built
    recreating-aldi-results/ // reproducing the ALDI-paper's results for different DAOD-methods w. the ALDI implementation framework on the respective dataset 
            aldi++-cityscapesrcnn-fpn-coco/
                cityscapes/
            at-cityscapesrcnn-fpn-coco/ // Adaptive Teacher
                ...        
``` 

