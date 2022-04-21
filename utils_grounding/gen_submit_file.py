import os
import json
import time
import copy
import argparse
from tqdm import tqdm

def load_refs(txt_db_path, split):
    reverie_id = {}
    refs_path = os.path.join(txt_db_path, f"reverie_{split}.db/refs.json")
    refs_data = json.load(open(refs_path, 'r'))
    for item in refs_data:
        id = item["id"]
        sent_id = item["sent_ids"][0]
        if id not in reverie_id:
            reverie_id[id] = []
        reverie_id[id].append(sent_id)
    return reverie_id

def get_obj_id(input_path, obj_path, txt_db_path, output_path):
    for root, dirs, files in os.walk(obj_path):
        for file in tqdm(files):
            obj_dir = os.path.join(obj_path, file)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if "seen" in file:
                obj_data = json.load(open(obj_dir, 'r'))
                input_dir = os.path.join(input_path, "submit_val_seen.json")
                submit_data = json.load(open(input_dir, 'r'))
                reverie_id = load_refs(txt_db_path, "val_seen")
                output_dir = os.path.join(output_path, "submit_val_seen.json")
            if "unseen" in file:
                obj_data = json.load(open(obj_dir, 'r'))
                input_dir = os.path.join(input_path, "submit_val_unseen.json")
                submit_data = json.load(open(input_dir, 'r'))
                reverie_id = load_refs(txt_db_path, "val_unseen")
                output_dir = os.path.join(output_path, "submit_val_unseen.json")
            if "test" in file:
                obj_data = json.load(open(obj_dir, 'r'))
                input_dir = os.path.join(input_path, "submit_test.json")
                submit_data = json.load(open(input_dir, 'r'))
                reverie_id = load_refs(txt_db_path, "test")
                output_dir = os.path.join(output_path, "submit_test.json")
            pred_id_view={}
            for item in reverie_id:
                for sent_id in reverie_id[item]:
                    sent_id = str(sent_id)
                    if item not in pred_id_view:
                        pred_id_view[item] = []
                    pred_id_view[item].append(obj_data[sent_id])
            pred_obj_vote={}
            for item_id in pred_id_view:
                for sent_id in pred_id_view[item_id]:
                    if item_id not in pred_obj_vote:
                        pred_obj_vote[item_id] = {}
                    if sent_id["object_id"] not in pred_obj_vote[item_id]:
                        pred_obj_vote[item_id][sent_id["object_id"]] = []
                        pred_obj_vote[item_id][sent_id["object_id"]].append(0) # num_obj
                        pred_obj_vote[item_id][sent_id["object_id"]].append(5.0) # conf>5
                    pred_obj_vote[item_id][sent_id["object_id"]][0]+=1
                    if sent_id["confs"]>pred_obj_vote[item_id][sent_id["object_id"]][1]:
                        pred_obj_vote[item_id][sent_id["object_id"]][1] = sent_id["confs"]
            pred_obj={}
            for item_id in pred_obj_vote:
                vote = 0
                conf = 0
                for obj in pred_obj_vote[item_id]:
                    if pred_obj_vote[item_id][obj][0] >= vote:
                        if pred_obj_vote[item_id][obj][0] > vote:
                            if pred_obj_vote[item_id][obj][1] > 5:
                                obj_id = obj
                                vote = pred_obj_vote[item_id][obj][0]
                                conf = pred_obj_vote[item_id][obj][1]
                        if pred_obj_vote[item_id][obj][0] == vote:
                            if pred_obj_vote[item_id][obj][1] > conf:
                                obj_id = obj
                                vote = pred_obj_vote[item_id][obj][0]
                                conf = pred_obj_vote[item_id][obj][1]
                if item_id not in pred_obj:
                    pred_obj[item_id] = []
                pred_obj[item_id].append(obj_id)
                pred_obj[item_id].append(vote)
                pred_obj[item_id].append(conf)
            new_submit_data=[]
            for item in submit_data:
                temp_item = copy.deepcopy(item)
                temp_item["predObjId"] = pred_obj[item["instr_id"]][0]
                new_submit_data.append(temp_item)
            with open(output_dir,'w') as f:
                json.dump(new_submit_data,f,indent=4)

def main(args):
    submit_input_path = "input_nav_dir"
    uniter_obj_path = "weights/results"
    txt_db_path = "txt_db/txt"
    get_obj_id(submit_input_path, uniter_obj_path, txt_db_path, args.output_dir)

if __name__ == "__main__":
    tic = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    main(args)
    toc = time.time()
    print("time:", toc - tic)


