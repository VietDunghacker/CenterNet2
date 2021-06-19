import itertools
import json
import os
import pandas as pd
import time
from typing import Any, Dict, List, Set

import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.engine import default_argument_parser, DefaultTrainer, DefaultPredictor, hooks, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.structures import BoxMode

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

classes = [
  'Candice Rene Accola',
  'Chelsea Royce Tavares',
  'Claire Rhiannon Holt',
  'Đỗ Viki',
  'Elizabeth Blackmore',
  'Elizabeth Melise Jow',
  'Emma Kristina Degerstedt',
  'Evanna Lynch',
  'Julianne Alexandra Hough',
  'Katerina Alexandre Hartford Graham',
  'Kayla Noelle Ewell',
  'Kelly Ann Hu',
  'Nikolina Kamenova Dobreva',
  'Odessa Zion Segall Adlon',
  'Penelope Mitchell',
  'Sara Canning',
  'Scarlett Hannah Byrne',
  'Teressa Liane',
]

def get_celebrity_dicts(csv_path):
    data_csv = pd.read_csv(csv_path)
    image_id_list = data_csv.image_id.unique()
    dataset_dicts = []
    for image_id in image_id_list:
        record = {}
        df = data_csv[data_csv['image_id'] == image_id].reset_index(drop = True)
        record["file_name"] = df["image_path"][0]
        record["image_id"] = image_id
        record["height"] = 800
        record["width"] = 800

        objs = []
        for idx, row in df.iterrows():
            x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

            obj = {
                    "bbox": [int(x_min * 800), int(y_min * 800), int(x_max * 800), int(y_max * 800)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": row["class_id"]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

class Trainer(DefaultTrainer):
#    """
#    Extension of the Trainer class adapted to SparseRCNN.
#    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, ("bbox", ), False, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        custom_augmentations = build_custom_augmentation(cfg, True)
        custom_augmentations.extend([
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
        ])
        mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else DatasetMapper(cfg, True, augmentations=custom_augmentations)
        if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            from centernet.data.custom_dataset_dataloader import  build_custom_train_loader
            return build_custom_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else DatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file("/content/CenterNet2/projects/CenterNet2/configs/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.yaml")
    cfg.DATASETS.TRAIN = ("celebrity_train",)
    cfg.DATASETS.TEST = ("celebrity_valid",)
    return cfg


def main(args):
    for d in ["train", "valid"]:
        DatasetCatalog.register("celebrity_" + d, lambda d=d: get_celebrity_dicts('/content/{}.csv'.format(d)))
        MetadataCatalog.get("celebrity_" + d).set(thing_classes=classes, evaluator_type = "coco")

    chest_metadata = MetadataCatalog.get("celebrity_train")

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
