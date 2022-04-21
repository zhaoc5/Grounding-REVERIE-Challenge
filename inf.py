import argparse
import json
import os
from os.path import exists
from time import time
import torch
from torch.utils.data import DataLoader
import numpy
from apex import amp
from horovod import torch as hvd
from cytoolz import concat
from data import (PrefetchLoader, DetectFeatLmdb, ReTxtTokLmdb,
                  ReEvalDataset, re_eval_collate)
from data.sampler import DistributedSampler
from model.re import UniterForReferringExpressionComprehension
from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import IMG_DIM

def write_to_tmp(txt, tmp_file):
    if tmp_file:
        f = open(tmp_file, "a")
        f.write(txt)

def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_epoch_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    model = UniterForReferringExpressionComprehension.from_pretrained(
        f'{opts.output_dir}/log/model.json', checkpoint,
        img_dim=IMG_DIM, mlp=1)
    model.to(device)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')
    img_db_type = "det"
    eval_img_db = DetectFeatLmdb(opts.img_db, -1, 100, 1, 100, False)
    txt_dbs = opts.txt_db.split(':')
    for txt_db in txt_dbs:
        print(f'Evaluating {txt_db}')
        eval_txt_db = ReTxtTokLmdb(txt_db, -1)
        eval_dataset = ReEvalDataset(eval_txt_db, eval_img_db, use_gt_feat=img_db_type == "gt")
        sampler = DistributedSampler(eval_dataset, num_replicas=n_gpu,
                                     rank=rank, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=sampler,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=re_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader)
        predictions  = evaluate(model, eval_dataloader)
        result_dir = f'{opts.output_dir}/results'
        results = all_gather_list(predictions)
        if not exists(result_dir):
            os.makedirs(result_dir)
        db_split = txt_db.split('/')[-1].split('.')[0]
        with open(f'{opts.output_dir}/results/'
              f'results_predObjId_{db_split}.json', 'w') as f:
            json.dump(results, f,indent=4)
        print(f'{opts.output_dir}/results')

@torch.no_grad()
def evaluate(model, eval_loader):
    LOGGER.info("start running evaluation...")
    model.eval()
    st = time()
    predictions = {}
    for i, batch in enumerate(eval_loader):
        (sent_ids,object_ids_list) = (batch['sent_ids'],batch['object_id'])
        scores = model(batch, compute_loss=False)
        confs, ixs = torch.max(scores, 1)
        ixs = ixs.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()
        for ix, conf, sent_id,object_ids in \
            zip(ixs, confs, sent_ids,object_ids_list):
            object_id = object_ids[ix]
            predictions[int(sent_id)] = {
                'pred_idx': int(ix),
                'object_id':int(object_id),
                'confs': float(conf)}
        if i % 10 == 0 and hvd.rank() == 0:
            n_results = len(predictions)
            n_results *= hvd.size()
            LOGGER.info(f'{n_results}/{len(eval_loader.dataset)} '
                        'answers predicted')
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="can be the path to binary or int number (step)")
    parser.add_argument("--batch_size",
                        default=512, type=int,
                        help="number of sentences per batch")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory of the training command")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=2,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--tmp_file', type=str, default=None,
                        help="write results to tmp file")
    args = parser.parse_args()
    main(args)
