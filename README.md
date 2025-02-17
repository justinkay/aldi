# File structure 

## Experiments and models 
```json
experiments // configs and output files for experiments we performed/models we built
    configs // config files (to recreate experiments)
        backbone/ // the basic model used as backbone, before any transfer learning/domain adaptation
            Base-RCNN-FPN.yaml // ResNet50 w. FPN pretrained on COCO
            ...
        baseline/ // Baseline/initialisation of the student/teacher models, i.e. burn-in checkpoints for the respective dataset (denoted in filename) - also serves as the config for the source-only model, essentially the backbone w. transfer learning on the source domain.
            Base-RCNN-FPN-Cityscapes...yaml
            ...
        daod/ // configs for models using a DOAD-method (specified by the subfolder name) - all set up w. the ALDI implementation framework. The filename specifies the burn-in model (baseline).
            aldi++/ 
                ALDI-Best-Cityscapes.yaml // ALDI++ w.  ResNet50 w. FPN w- COCO-pretraining as backbone
                ...
            at/ // Adaptive Teacher
                ...
            ...
        oracle/ // Oracle-model - trained solely on (labelled) target-data, it's the backbone (specified in file name) w. transfer learning on the target domain
            OracleT-RCNN-FPN-Cityscapes...yaml
            ...
    output/ // selected files from the produced output from running train_net on the respective experiment (denoted by folder name)
        reproductions/ // reproducing the ALDI-paper's results for different DAOD-methods (folder name) w. the ALDI implementation framework on the respective dataset (subfolder name)
            aldi++/
                cityscapes/
                    ... // the produced config (frankenstein of everything), log, and the best models
            at/ // Adaptive Teacher
                ...
        
```


experiments
    backbone-configs
        ...

    experiment-name // e.g. the DAOD-method and the datasets
        configs/
        models/
        output/
    recreating-aldi-results
        aldi++-cityscapes // case of experiment, e.g. the DAOD method and the data set
            configs/
                ...
            models/ 
                ...
            output/ <-- ignore!


Common-configs contains the following types of models, which are generally reused in the experiments. The baseline and oracle 
backbone =  basic backbone model, like ResNet50 w. RPN - pre-trained on COCO or 
            imagenet or whatever
baseline =  the backbone w. transfer learning on SOURCE data - this is both the source-only 
            model AND the burn-in model (initialisation of student/teacher for the DAOD-method)
oracle =    the backebone w. transfer learning on TARGET data


are generally reused, so they just sit in the outer folder, backbone-configs

