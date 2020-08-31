import comet_ml

import os
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from typing import Any

from genomics_data import RandomDataIteratorOneSeq, SequentialDataIterator, DatasetPL
from genomics_utils import available, ensure_directories, boolean_string


def lightning_train(trainer: pl.Trainer,
                    model: pl.LightningModule,
                    data_module: pl.LightningDataModule,
                    output: str,
                    resume: bool
                    ):
    
    model_root, = ensure_directories(output, 'models/')
    parameters_path = os.path.join(
        model_root,
        '{model}.pt'.format(model=model.name)
    )

    if resume:
        trainer = pl.Trainer(resume_from_checkpoint=parameters_path)

    trainer.logger.experiment.set_name(model.name)
    print("Running {}-model...".format(model.name))
    
    # main part here
    data_module.setup('fit')
    trainer.fit(model, data_module)
    
    model.save(trainer, parameters_path)
    # logger.log_losses("dataset", model.name, losses)


def lightning_test(trainer: pl.Trainer,
                   model: pl.LightningModule,
                   data_module: pl.LightningDataModule
                   ):
    model_root, = ensure_directories(args.output, 'models/')
    parameters_path = os.path.join(
        model_root,
        '{model}.pt'.format(model=model.name)
    )
    print(type(model))
    model = type(model).load_from_checkpoint(checkpoint_path=parameters_path)
    
    data_module.setup('test')
    trainer.test(model, data_module)
    # logger.log_coalescent_heatmap(model.name, heatmap_preds, "00000")


if __name__ == '__main__':
    import argparse
    from parser_args import gru_add_arguments, conv_bert_add_arguments, bert_add_arguments, conv_add_arguments, \
        gru_one_dir_add_arguments
    
    parser = argparse.ArgumentParser(prog='Genomics')
    parser.add_argument(
        '--data', type=str, default='data/micro_data/',
        help='directory from which data is read or to which data will be downloaded if absent, '
    )
    parser.add_argument('--resume', type=boolean_string, default=False)
    parser.add_argument('--output', type=str, default='output/', help='root directory to write various statistics to')
    parser.add_argument('--device', type=str, default='cpu', help='device in torch format')
    parser.add_argument('--logger', type=str, choices=['local', 'comet'], default='local')
    parser.add_argument('--project', type=str, default=None, help='project name for comet logger, None by default')
    parser.add_argument('--workspace', type=str, default=None, help='workspace for comet logger, None by default')
    parser.add_argument('--offline', type=boolean_string, default=True, help='logger mode')
    parser.add_argument('--quiet', type=boolean_string, default=False)
    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--n_token_in', type=int, default=2)
    parser.add_argument('--n_output', type=int, default=20)
    
    parser.add_argument('--tr_file_first', type=int, default=0)
    parser.add_argument('--tr_file_last', type=int, default=0)
    parser.add_argument('--te_file_first', type=int, default=1)
    parser.add_argument('--te_file_last', type=int, default=1)
    
    parser.add_argument('--action', type=str, choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--auto_lr_find', type=boolean_string, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--shuffle', type=boolean_string, default=False)
    
    model_parsers = parser.add_subparsers(title='models', description='model to choose', dest='model')
    
    gru_parser = model_parsers.add_parser('gru')
    conv_parser = model_parsers.add_parser('conv')
    conv_gru_parser = model_parsers.add_parser('conv-gru')
    bert_parser = model_parsers.add_parser('bert')
    conv_bert_parser = model_parsers.add_parser('conv-bert')
    
    gru_add_arguments(gru_parser)
    conv_add_arguments(conv_parser)
    bert_add_arguments(bert_parser)
    conv_bert_add_arguments(conv_bert_parser)
    
    args = parser.parse_args()
    print(args)
    
    model = available.models[args.model].Model(args)
    
    comet_logger = pl_loggers.CometLogger(workspace=args.workspace,
                                          project_name=args.project,
                                          save_dir=args.output,
                                          offline=args.offline,
                                          parse_args=False,
                                          auto_metric_logging=False,
                                          disabled=True
                                          )
    
    trainer = pl.Trainer(logger=comet_logger,
                         max_epochs=args.epochs,
                         auto_lr_find=args.auto_lr_find,
                         checkpoint_callback=False
                         )
    
    data_module = DatasetPL(path=args.data,
                            tr_file_first=args.tr_file_first,
                            tr_file_last=args.tr_file_last,
                            te_file_first=args.te_file_first,
                            te_file_last=args.te_file_last,
                            one_side_padding=args.padding,
                            seq_len=args.seq_len,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_workers)
    
    if args.action == 'train':
        lightning_train(trainer=trainer,
                        model=model,
                        data_module=data_module,
                        output=args.output,
                        resume=args.resume
                        )
        # train(
        #     dataset=dataset, model=model,
        #     device=args.device, seed=args.seed,
        #     n_epochs=args.epochs, batch_size=args.batch_size,
        #     output_path=args.output, logger=logger,
        #     num_workers=args.num_workers, quiet=args.quiet
        # )
        # test(
        #     dataset=dataset, model=model,
        #     device=args.device, batch_size=args.batch_size,
        #     output_path=args.output, logger=logger, project=args.project,
        #     workspace=args.workspace
        # )
    elif args.action == 'test':
        lightning_test(trainer=trainer,
                       model=model,
                       data_module=data_module
                       )
        # test(
        #     dataset=dataset, model=model,
        #     device=args.device, batch_size=args.batch_size,
        #     output_path=args.output, logger=logger, project=args.project,
        #     workspace=args.workspace
        # )
    else:
        raise ValueError("Unknown option {}".format(args.action))
