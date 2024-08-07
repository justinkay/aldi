#!/bin/bash

#OVERRIDE='DATASETS.TRAIN ("squidle_urchin_2009_train",) DATASETS.TEST ("squidle_urchin_2011_test",) DATASETS.UNLABELED ("squidle_urchin_2011_test",)'
OVERRIDE=""
GROUP_TAGS='LOGGING.GROUP_TAGS "Inf2UDD,S2"'
echo $OVERRIDE $GROUP_TAGS
#python tools/train_net.py --config configs/urchininf/Base-RCNN-FPN-urchininf_weakaug.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml  ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/MeanTeacher-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/urchininf_priorart/AT-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/ALDI-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/urchininf_priorart/MIC-urchininf_NoBurnIn.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/urchininf_priorart/SADA_urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/urchininf_baseline/Probabilistic-RCNN-FPN-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}
python tools/train_net.py --config configs/urchininf/urchininf_priorart/ProbabilisticTeacher-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}

