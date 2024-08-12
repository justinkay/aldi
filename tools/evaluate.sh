#!/bin/bash

#OVERRIDE='DATASETS.TRAIN ("squidle_urchin_2009_train",) DATASETS.TEST ("squidle_urchin_2011_test",) DATASETS.UNLABELED ("squidle_urchin_2011_test",)'
OVERRIDE='DATASETS.TEST ("SUODAC2020_test",) '
#OVERRIDE=""
GROUP_TAGS='LOGGING.GROUP_TAGS "Inf2UDD,TEST"'
echo $OVERRIDE $GROUP_TAGS
python tools/train_net.py --eval-only --config configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_base_strongaug_ema/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --eval-only --config configs/urchininf/MeanTeacher-urchininf.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_MT/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --eval-only --config configs/urchininf/urchininf_priorart/AT-urchininf.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_AT/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --eval-only --config configs/urchininf/urchininf_priorart/SADA_urchininf.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_SADA/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}


#python tools/train_net.py --eval-only --config configs/urchininf/Base-RCNN-FPN-urchininf_weakaug.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_base_weakaug/model_final.pth'  ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --eval-only --config configs/urchininf/ALDI-urchininf.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_ALDI/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --eval-only --config configs/urchininf/OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_oracle_strongaug_ema/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --eval-only --config configs/urchininf/urchininf_priorart/MIC-urchininf_NoBurnIn.yaml MODEL.WEIGHTS 'outputs/urchininf/urchininf_MIC/model_final.pth' ${OVERRIDE} ${GROUP_TAGS}



