import argparse
import json
import pickle
import os
from os.path import exists
from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from data.data import open_lmdb

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

def process_referring_expressions(refs, instances, iid_to_ann_ids, db, tokenizer, split):
    print(split)
    image_set = set([ref['image_id'] for ref in refs if ref['split'] == split])
    images = []
    for img in instances['images']:
        if img['id'] in image_set:
            images.append({
                'id': img['id'],
                'file_name': img['file_name'],
                'ann_ids': iid_to_ann_ids[str(img['id'])],
                'height': img['height'], 
                'width': img['width']})
    annotations = []
    for ann in instances['annotations']:
        if ann['image_id'] in image_set:
            annotations.append({
                'id': ann['id'], 
                'area': ann['area'], 
                'bbox': ann['bbox'],
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'iscrowd': ann['iscrowd']
            })
    categories = instances['categories']
    refs = [ref for ref in refs if ref['split'] == split]
    print(f"Processing {len(refs)} annotations...")
    id2len = {}
    for ref in tqdm(refs, desc='processing referring expressions'):
        reverie_id = ref['id']
        ref_id = ref['ref_id']
        image_id = ref['image_id']
        h5_name = ref['h5_name']
        for sent in ref['sentences']:
            sent_id = sent['sent_id']
            input_ids = tokenizer(sent['sent'])
            id2len[str(sent_id)] = len(input_ids)
            db[str(sent_id)] = {
                'reverie_id':reverie_id,
                'sent_id': sent_id, 
                'sent': sent['sent'],
                'ref_id': ref_id, 
                'image_id': image_id, 
                'h5_name': h5_name, 
                'input_ids': input_ids
            }
    return id2len, images, annotations, categories, refs

def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0], len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)
    open_db = curry(open_lmdb, opts.output, readonly=False)
    output_field_name = ['id2len', 'txt2img']
    with open_db() as db:
        if opts.task == 're':
            print(opts.annotations[0])
            print(opts.annotations[1])
            print(opts.annotations[2])
            data = pickle.load(open(opts.annotations[0], 'rb'))
            instances = json.load(open(opts.annotations[1], 'r'))
            iid_to_ann_ids = json.load(open(opts.annotations[2], 'r'))['iid_to_ann_ids']
            basename = os.path.basename(opts.output)
            index_first_underline = basename.index("_") + 1
            index_second_underline = basename.index(".")
            img_split = basename[index_first_underline:index_second_underline]
            jsons = process_referring_expressions(
                data, instances, iid_to_ann_ids,
                db, tokenizer, img_split)
            output_field_name = ['id2len', 'images', 'annotations', 'categories', 'refs']
    for dump, name in zip(jsons, output_field_name):
        with open(f'{opts.output}/{name}.json', 'w') as f:
            json.dump(dump, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, nargs='+',
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--task', default='re')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)
