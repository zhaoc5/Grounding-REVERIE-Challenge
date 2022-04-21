import os
import time
import argparse
from os.path import basename, exists
import h5py
from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb
import json
import glob
import pickle as pkl
import msgpack
import msgpack_numpy
import multiprocessing as mp
msgpack_numpy.patch()

def to_norm_bb(bboxes):
    image_w=640
    image_h=480
    box_width = bboxes[2] - bboxes[0]
    box_height = bboxes[3] - bboxes[1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h
    scaled_x = bboxes[0] / image_w
    scaled_y = bboxes[1] / image_h
    box_width = box_width[..., np.newaxis]
    box_height = box_height[..., np.newaxis]
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]
    normalized_bbox = np.concatenate((scaled_x, scaled_y,
                                      scaled_x + scaled_width,
                                      scaled_y + scaled_height,
                                      scaled_width, scaled_height), axis=0)
    return normalized_bbox

@curry
def load_h5(fname):
    img_dump = h5py.File(fname, 'r')
    dump = {}
    dump['box_id'] = img_dump['box_id'][()].astype(np.float32)
    dump['features'] = img_dump['features'][()].astype(np.float32)
    dump['norm_bb'] = to_norm_bb(img_dump['bbox'][()]).astype(np.float32)
    name = basename(fname)
    return name, dump

def to_mdb(data):
    dump = {}
    box_id=[]
    features=[]
    norm_bb = []
    data = {int(k) : v for k, v in data.items()}
    data = sorted(data.items(), key=lambda t: t[0])
    for obj in data:
        box_id.append(int(obj[0]))
        features.append(obj[1]['features'].toarray().squeeze())
        norm_bb.append(to_norm_bb(obj[1]['boxes'].toarray().squeeze()))
    dump['box_id'] = np.array(box_id).astype(np.float16)
    dump['features'] = np.array(features).astype(np.float16)
    dump['norm_bb'] = np.array(norm_bb).astype(np.float16)
    return dump

def main_pkl(args):
    db_name = f'uniter-reverie'
    if not exists(f'{args.output}'):
        os.makedirs(f'{args.output}')
    env = lmdb.open(f'{args.output}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    with open(args.feats_path, 'rb') as f:
        print("loading pkl...")
        data = pkl.load(f)
    for scan in tqdm(data):
        for vp in data[scan]:
            for view in data[scan][vp]:
                item = to_mdb(data[scan][vp][view])
                fname = f"{scan}_{vp}_{int(view):02}.h5"
                txn.put(key=fname.encode('utf-8'), value=msgpack.dumps(item, use_bin_type=True))
            txn.commit()
            txn = env.begin(write=True)
    env.close()
    
def main_h5(opts):
    if opts.feats_path[-1] == '/':
        opts.feats_path = opts.feats_path[:-1]
    db_name = f'uniter-reverie'
    if not exists(f'{opts.output}'):
        os.makedirs(f'{opts.output}')
    env = lmdb.open(f'{opts.output}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    files = glob.glob(f'{opts.feats_path}/*.h5')
    load = load_h5()
    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features) in enumerate(pool.imap_unordered(load, files, chunksize=128)):
            if not features:
                continue
            dump = msgpack.dumps(features, use_bin_type=True)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            pbar.update(1)
        txn.put(key=b'__keys__',value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()

if __name__ == '__main__':
    tic = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--feats_path')
    parser.add_argument('--feats_format')
    args = parser.parse_args()
    if args.feats_format == 'pkl':
        main_pkl(args)
    elif args.feats_format == 'h5':
        main_h5(args)
    else:
        print('ERROR')
    toc = time.time()
    print("time:", toc - tic)


