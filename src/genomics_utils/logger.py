import os
import json

import numpy as np
from comet_ml import Experiment as CometExperiment
import matplotlib.pyplot as plt

__all__ = [
	'LocalLogger', 'CometLogger',
	'get_logger',
]


def warn():
	import traceback
	import warnings
	warnings.warn(traceback.format_exc())


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


class Logger(object):
	def log_metrics(self, dataset_name, model_name, **kwargs):
		raise NotImplementedError()
	
	def log_losses(self, dataset_name, model_name, losses):
		raise NotImplementedError()


class LocalLogger(Logger):
	"""
    Writing json logger
    """
	
	def __init__(self, root):
		from .common import ensure_directories
		self._report_root, self._figure_root = ensure_directories(root, 'reports/', 'figures/')
		
		super(LocalLogger, self).__init__()
	
	def log_metrics(self, dataset_name, model_name, **info):
		path = os.path.join(
			self._report_root,
			'{dataset}-{model}.json'.format(dataset=dataset_name, model=model_name)
		)
		
		info['dataset'] = dataset_name
		info['model'] = model_name
		with open(path, 'w') as f:
			json.dump(info, f, indent=2)
	
	def _log_learning_curve(self, dataset_name, model_name, losses):
		from .viz import make_learning_curve
		
		f = make_learning_curve(dataset_name, model_name, losses)
		plt.savefig(
			os.path.join(
				self._figure_root,
				'{dataset}-{model}.png'.format(dataset=dataset_name, model=model_name)
			)
		)
		return f
	
	def log_losses(self, dataset_name, model_name, losses):
		f = self._log_learning_curve(dataset_name, model_name, losses)
		plt.close(f)


class CometLogger(LocalLogger):
	"""
    Comet ml logger
    """
	
	def __init__(self, root, experiment: CometExperiment):
		self._experiment = experiment
		
		super(CometLogger, self).__init__(root)
	
	def log_metrics(self, dataset_name, model_name, **info):
		super(CometLogger, self).log_metrics(dataset_name, model_name, **info)
		
		for metric_name, value in info.items():
			self._experiment.log_metric(
				'{dataset}_{model}_{metric}'.format(dataset=dataset_name, model=model_name, metric=metric_name),
				value
			)
	
	def log_losses(self, dataset_name, model_name, losses):
		f = self._log_learning_curve(dataset_name, model_name, losses)
		self._experiment.log_figure(
			"Losses-{}".format(self._experiment.project_name),
			f
		)
		plt.close(f)


def get_logger(logger, root, project=None, workspace=None) -> Logger:
	from genomics_utils import LocalLogger, CometLogger
	
	if logger.lower() == "local":
		return LocalLogger(root)
	
	elif logger.lower() == "comet":
		assert project is not None, 'for comet logger, please, provide project name'
		assert workspace is not None, 'for comet logger, please, provide workspace'
		
		experiment = CometExperiment(project_name=project, workspace=workspace)
		return CometLogger(root=root, experiment=experiment)
	
	else:
		raise ValueError("Unknown experiment context")
