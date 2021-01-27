import comet_ml
from typing import Any
import os

import pytorch_lightning as pl

from genomics_data import DatasetPL
from genomics_utils import available, ensure_directories, boolean_string, float_to_int
from genomics_utils import CometLightningLogger, ExistingCometLightningLogger


def lightning_train(trainer: pl.Trainer,
                    model: pl.LightningModule,
                    data_module: pl.LightningDataModule,
                    checkpoint_path: str,
                    resume: bool
                    ):
    if resume:
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path)
    
    exp_key = trainer.logger.experiment.get_key()
    print("Running {}-model...".format(model.name))
    
    # main part here
    data_module.setup('fit')
    trainer.fit(model=model, datamodule=data_module)
    
    model.save(trainer, checkpoint_path)
    
    return trainer, model, exp_key


def lightning_test(trainer: pl.Trainer,
                   model: pl.LightningModule,
                   checkpoint_path: str,
                   datamodule: pl.LightningDataModule,
                   experiment_key: str,
                   logger: Any
                   ):
    if len(experiment_key) > 0:
        logger = CometLightningLogger(experiment_key=experiment_key,
                                      experiment_name=model.name)
    
    # trainer = pl.Trainer(logger=logger)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    
    datamodule.setup('test')
    trainer.test(model=model,
                 datamodule=datamodule)
    
    return


if __name__ == '__main__':
    import argparse
    from parser_args import gru_add_arguments, gru_fg_add_arguments, conv_bert_add_arguments, bert_add_arguments, \
        conv_add_arguments, \
        gru_one_dir_add_arguments, conv_gru_add_arguments
    
    parser = argparse.ArgumentParser(prog='Genomics')
    parser.add_argument(
        '--data', type=str, default='data/micro_data/',
        help='directory from which data is read or to which data will be downloaded if absent, '
    )
    parser.add_argument('--resume', type=boolean_string, default=False)
    parser.add_argument('--exp_key', type=str, default="")
    parser.add_argument('--output', type=str, default='output/', help='root directory to write various statistics to')
    parser.add_argument('--device', type=str, default='cpu', help='device in torch format')
    parser.add_argument('--logger', type=str, choices=['local', 'comet'], default='local')
    parser.add_argument('--cmt_project', type=str, default=None, help='project name for comet logger, None by default')
    parser.add_argument('--cmt_workspace', type=str, default=None, help='workspace for comet logger, None by default')
    parser.add_argument('--cmt_offline', type=boolean_string, default=True, help='logger mode')
    parser.add_argument('--cmt_disabled', type=boolean_string, default=True)
    parser.add_argument('--quiet', type=boolean_string, default=False)
    
    parser.add_argument('--sqz', type=boolean_string, default=False)
    parser.add_argument('--seq_len', type=float_to_int, default=1)
    parser.add_argument('--sqz_seq_len', type=float_to_int, default=1)
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
    parser.add_argument('--auto_lr_find', type=boolean_string, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--shuffle', type=boolean_string, default=False)
    
    model_parsers = parser.add_subparsers(title='models', description='model to choose', dest='model')
    
    gru_parser = model_parsers.add_parser('gru')
    gru_fg_parser = model_parsers.add_parser('gru_fg')
    conv_gru_parser = model_parsers.add_parser('conv_gru')
    conv_parser = model_parsers.add_parser('conv')
    bert_parser = model_parsers.add_parser('bert')
    conv_bert_parser = model_parsers.add_parser('conv_bert')
    
    gru_add_arguments(gru_parser)
    gru_fg_add_arguments(gru_fg_parser)
    conv_gru_add_arguments(conv_gru_parser)
    conv_add_arguments(conv_parser)
    bert_add_arguments(bert_parser)
    conv_bert_add_arguments(conv_bert_parser)
    
    args = parser.parse_args()
    print(args)
    print(available.models)
    model = available.models[args.model].Model(args)
    
    model_root, = ensure_directories(args.output, 'models/')
    default_root_dir, = ensure_directories(args.output, 'models/{}'.format(model.name))
    
    checkpoint_path = os.path.join(
        default_root_dir,
        '{model}.pt'.format(model=model.name)
    )
    
    comet_logger = CometLightningLogger(workspace=args.cmt_workspace,
                                        project_name=args.cmt_project,
                                        save_dir=default_root_dir,
                                        offline=args.cmt_offline,
                                        parse_args=False,
                                        auto_metric_logging=False,
                                        disabled=args.cmt_disabled,
                                        experiment_name=model.name
                                        )
    
    trainer = pl.Trainer(default_root_dir=default_root_dir,
                         logger=comet_logger,
                         max_epochs=args.epochs,
                         auto_lr_find=args.auto_lr_find,
                         checkpoint_callback=False,
                         gpus=0 if args.device == 'cpu' else 1,
                         # truncated_bptt_steps=None if args.truncated_bptt_steps == -1 else args.truncated_bptt_steps
                         )
    
    datamodule = DatasetPL(path=args.data,
                           tr_file_first=args.tr_file_first,
                           tr_file_last=args.tr_file_last,
                           te_file_first=args.te_file_first,
                           te_file_last=args.te_file_last,
                           one_side_padding=args.padding,
                           seq_len=args.seq_len,
                           sqz_seq_len=args.sqz_seq_len,
                           batch_size=args.batch_size,
                           n_output=args.n_output,
                           shuffle=args.shuffle,
                           num_workers=args.num_workers,
                           use_distance=args.sqz)
    
    if args.action == 'train':
        trainer, model, exp_key = lightning_train(trainer=trainer,
                                                  model=model,
                                                  data_module=datamodule,
                                                  checkpoint_path=checkpoint_path,
                                                  resume=args.resume
                                                  )
        # lightning_test(trainer=trainer,
        #                model=model,
        #                checkpoint_path=checkpoint_path,
        #                datamodule=datamodule,
        #                experiment_key="",
        #                logger=comet_logger,
        #                )
    elif args.action == 'test':
        lightning_test(trainer=trainer,
                       model=model,
                       checkpoint_path=checkpoint_path,
                       datamodule=datamodule,
                       experiment_key=args.exp_key,
                       logger=comet_logger
                       )
    else:
        raise ValueError("Unknown option {}".format(args.action))
