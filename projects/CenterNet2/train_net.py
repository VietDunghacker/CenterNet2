from collections import OrderedDict
import copy
import datetime
import itertools
import json
import logging
import numpy as np
import os
import pandas as pd
import time
from typing import Any, Dict, List, Set

import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.engine import default_argument_parser, default_setup, DefaultTrainer, DefaultPredictor, hooks, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, LVISEvaluator, inference_on_dataset, print_csv_format
from detectron2.modeling import build_model
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures import BoxMode
import detectron2.utils.comm as comm
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger

from sparsercnn import SparseRCNNDatasetMapper, add_sparsercnn_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

from fvcore.common.timer import Timer

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
  'Lý Thất Hi',
  'Nại Nại Fox',
  'Nikolina Kamenova Dobreva',
  'Odessa Zion Segall Adlon',
  'Penelope Mitchell',
  'Sara Canning',
  'Scarlett Hannah Byrne',
  'Teressa Liane',
  'Tô Tiểu Tiểu',
  'Vũ Trụ'
]

logger = logging.getLogger("detectron2")

'''class CustomMapper()
	def __init__(self, is_train: bool, augmentations: List[Union[T.Augmentation, T.Transform]]):
		self.is_train = is_train
		self.augmentations = T.AugmentationList(augmentations)'''

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

def do_test(cfg, model):
	results = OrderedDict()
	for dataset_name in cfg.DATASETS.TEST:
		mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else DatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
		data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
		output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
		evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

		if evaluator_type == "lvis":
			evaluator = LVISEvaluator(dataset_name, cfg, False, output_folder)
		elif evaluator_type == 'coco':
			evaluator = COCOEvaluator(dataset_name, ("bbox", ), False, output_folder)
		else:
			assert 0, evaluator_type
			
		results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)
		if comm.is_main_process():
			logger.info("Evaluation results for {} in csv format:".format(dataset_name))
			print_csv_format(results[dataset_name])
	if len(results) == 1:
		results = list(results.values())[0]
	return results

def do_train(cfg, model, resume=False):
	model.train()
	optimizer = build_optimizer(cfg, model)
	scheduler = build_lr_scheduler(cfg, optimizer)

	checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

	start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume,).get("iteration", -1) + 1)
	if cfg.SOLVER.RESET_ITER:
		logger.info('Reset loaded iteration. Start training from iteration 0.')
		start_iter = 0
	max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

	periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

	writers = (
		[
			CommonMetricPrinter(max_iter),
			JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
			TensorboardXWriter(cfg.OUTPUT_DIR),
		]
		if comm.is_main_process()
		else []
	)

	custom_augmentations = build_custom_augmentation(cfg, True)
	custom_augmentations.extend([
		T.RandomBrightness(0.8, 1.2),
		T.RandomContrast(0.8, 1.2),
		T.RandomSaturation(0.8, 1.2),
	])
	mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else DatasetMapper(cfg, True, augmentations=custom_augmentations)
	if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
		data_loader = build_detection_train_loader(cfg, mapper=mapper)
	else:
		from centernet.data.custom_dataset_dataloader import build_custom_train_loader
		data_loader = build_custom_train_loader(cfg, mapper=mapper)

	if cfg.DATALOADER.SAMPLER_TRAIN == "ClassAwareSampler":
		cw = copy.deepcopy(data_loader.batch_sampler.sampler.cw)
		txt = "Initial weight:\n"
		for i, name in enumerate(classes):
			txt += "{:40s}: {:6f}\n".format(name, cw[i])
		logger.info(txt)
		last_map = 0.

	with EventStorage(start_iter) as storage:
		step_timer = Timer()
		data_timer = Timer()
		start_time = time.perf_counter()
		for data, iteration in zip(data_loader, range(start_iter, max_iter)):
			data_time = data_timer.seconds()
			storage.put_scalars(data_time=data_time)
			step_timer.reset()
			loss_dict = model(data)

			losses = sum(loss for k, loss in loss_dict.items())
			assert torch.isfinite(losses).all(), loss_dict

			loss_dict_reduced = {k: v.item() \
				for k, v in comm.reduce_dict(loss_dict).items()}
			losses_reduced = sum(loss for loss in loss_dict_reduced.values())
			if comm.is_main_process():
				storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()

			storage.put_scalar(
				"lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

			step_time = step_timer.seconds()
			storage.put_scalars(time=step_time)
			data_timer.reset()
			scheduler.step()

			if (cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter):
				model.eval()
				test_result = do_test(cfg, model)
				model.train()

				if cfg.DATALOADER.SAMPLER_TRAIN == "ClassAwareSampler":
					maps = []
					for name in classes:
						maps.append(test_result['bbox']['AP-{}'.format(name)])
					maps = np.array(maps)

					data_loader.batch_sampler.sampler.cw = cw * ((1 - maps / 100 + 1e-6) ** 2)
					data_loader.batch_sampler.sampler.cw /= sum(data_loader.batch_sampler.sampler.cw)

					txt = "New weight:\n"
					for i, name in enumerate(classes):
						txt += "{:40s}: {:6f}\n".format(name, data_loader.batch_sampler.sampler.cw[i])
					logger.info(txt)

				comm.synchronize()

			if iteration - start_iter > 5 and ((iteration + 1) % 10 == 0 or iteration == max_iter):
				for writer in writers:
					writer.write()
			periodic_checkpointer.step(iteration)

			iteration = iteration + 1
			storage.step()

		total_time = time.perf_counter() - start_time
		logger.info(
			"Total training time: {}".format(
				str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
	"""
	Create configs and perform basic setups.
	"""
	cfg = get_cfg()
	add_sparsercnn_config(cfg)
	cfg.merge_from_file("/content/CenterNet2/projects/CenterNet2/configs/sparse.yaml")
	cfg.DATASETS.TRAIN = ("celebrity_train",)
	cfg.DATASETS.TEST = ("celebrity_valid",)
	cfg.freeze()
	default_setup(cfg, args)
	return cfg


def main(args):
	for d in ["train", "valid"]:
		DatasetCatalog.register("celebrity_" + d, lambda d=d: get_celebrity_dicts('/content/{}.csv'.format(d)))
		MetadataCatalog.get("celebrity_" + d).set(thing_classes=classes, evaluator_type = "coco")

	cfg = setup(args)

	model = build_model(cfg)

	if args.eval_only:
		DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
		if cfg.TEST.AUG.ENABLED:
			logger.info("Running inference with test-time augmentation ...")
			model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
		model.eval()
		return do_test(cfg, model)

	do_train(cfg, model, resume=args.resume)
	return do_test(cfg, model)


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
