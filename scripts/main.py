import comet_ml

import os
import torch
from torch.utils import data
import numpy as np

from genomics import Classifier
from genomics_data import OneSequenceDataset, Dataset
from genomics_utils import available, get_logger, ensure_directories

from tqdm import tqdm


def train(dataset, model, device, seed, n_epochs, batch_size,
		  output_path, logger, num_workers=1, quiet=False):
	model_root, = ensure_directories(output_path, 'models/')
	parameters_path = os.path.join(
		model_root,
		'{model}.pt'.format(model=model.name)
	)
	device = torch.device(device)
	torch.manual_seed(seed)
	
	params = {'batch_size': batch_size,
			  'num_workers': num_workers,
			  'shuffle': False
			  }
	
	train_loader = data.DataLoader(dataset.train_set, **params)
	clf = Classifier(model, device=device)
	if not quiet:
		print("Running {}-model...".format(model.name))
	losses = clf.fit(train_loader, n_epochs=n_epochs, progress=None if quiet else tqdm)
	
	clf.save(parameters_path, quiet)
	
	logger.log_losses("", model.name, losses)


def test(dataset, model, device, batch_size,
		 output_path, logger, project, workspace):
	log = get_logger(logger, output_path, project, workspace)
	
	model_root, = ensure_directories(output_path, 'models/')
	parameters_path = os.path.join(
		model_root,
		'{model}.pt'.format(model=model.name)
	)
	
	device = torch.device(device)
	
	train_loader = torch.utils.data.DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset.test_set, batch_size=batch_size, shuffle=True)
	
	clf = Classifier(model, device=device)
	
	state_dict = torch.load(parameters_path)
	clf.classifier.load_state_dict(state_dict)
	
	predictions_train, true_train = clf.predict(train_loader)
	predictions_test, true_test = clf.predict(test_loader)
	
	accuracy_train = np.mean(np.argmax(predictions_train, axis=1) == true_train)
	accuracy_test = np.mean(np.argmax(predictions_test, axis=1) == true_test)
	
	log.log_metrics("d", model.name, accuracy_train=accuracy_train, accuracy_test=accuracy_test)


if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(prog='Genomics')
	parser.add_argument(
		'--data', type=str, default='data/micro_data/',
		help='directory from which data is read or to which data will be downloaded if absent, '
	)
	parser.add_argument('--output', type=str, default='output/', help='root directory to write various statistics to')
	parser.add_argument('--device', type=str, default='cpu', help='device in torch format')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--logger', type=str, choices=['local', 'comet'], default='local')
	parser.add_argument('--project', type=str, default=None, help='project name for comet logger, None by default')
	parser.add_argument('--workspace', type=str, default=None, help='workspace for comet logger, None by default')
	parser.add_argument('--quiet', type=bool, default=False)
	parser.add_argument('--seq_len', type=int, default=1)
	parser.add_argument('--tgt_len', type=int, default=1)
	parser.add_argument('--n_output', type=int, default=20)
	
	parser.add_argument('--action', type=str, choices=['train', 'test'])
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--epochs', type=int, default=16)
	parser.add_argument('--lr', type=float, default=0.001)
	
	model_parsers = parser.add_subparsers(title='models', description='model to choose', dest='model')
	gru_parser = model_parsers.add_parser('gru')
	gru_parser.add_argument('--input_size', type=int, default=1)
	gru_parser.add_argument('--hidden_size', type=int, default=64)
	gru_parser.add_argument('--num_layers', type=int, default=2)
	gru_parser.add_argument('--batch_first', type=bool, default=True)
	gru_parser.add_argument('--bidirectional', type=bool, default=True)
	gru_parser.add_argument('--dropout', type=float, default=0.1)
	
	# linear_parser = model_parsers.add_parser("linear")
	# linear_parser.add_argument()
	
	args = parser.parse_args()
	print(args)
	
	logger = get_logger(args.logger, args.output, project=args.project, workspace=args.workspace)
	
	model = available.models[args.model].Model(args)
	assert (args.seq_len - args.tgt_len) % 2 == 0
	dataset = OneSequenceDataset(args.data, args.tgt_len, int((args.seq_len - args.tgt_len) / 2))
	
	if args.action == 'train':
		train(
			dataset=dataset, model=model,
			device=args.device, seed=args.seed,
			n_epochs=args.epochs, batch_size=args.batch_size,
			output_path=args.output, logger=logger,
			num_workers=args.num_workers, quiet=args.quiet
		)
	elif args.action == 'test':
		test(
			dataset=dataset, model=model,
			device=args.device, batch_size=args.batch_size,
			output_path=args.output, logger=args.logger, project=args.project,
			workspace=args.workspace
		)
	else:
		raise ValueError("Unknown option {}".format(args.action))
