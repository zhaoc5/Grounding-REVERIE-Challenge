import os
import argparse
import json
import time
import copy
from tqdm import tqdm

def nav_to_reverie(reverie, nav, uniter_reverie):
    test = False
    if "submit_test" in nav:
        test = True
    reverie_data = json.load(open(reverie, 'r'))
    nav_data = json.load(open(nav, 'r'))
    uniter_reverie_data = []
    for nav_item in tqdm(nav_data):
        target_vp = nav_item["trajectory"][-1][0]
        if test:
            id, sent_idx = nav_item["instr_id"].split('_')
        else:
            id_0, id_1, sent_idx = nav_item["instr_id"].split('_')
            id = id_0+"_"+id_1
        for reverie_item in reverie_data:
            if id == reverie_item["id"]:
                temp_reverie_item = copy.deepcopy(reverie_item)
                temp_reverie_item["instructions"] = []
                temp_reverie_item["instructions"].append(reverie_item["instructions"][int(sent_idx)])
                temp_reverie_item["id"] = nav_item["instr_id"]
                temp_reverie_item["path"] = []
                temp_reverie_item["path"].append(target_vp)
                if temp_reverie_item not in uniter_reverie_data:
                    uniter_reverie_data.append(temp_reverie_item)
    with open(uniter_reverie,'w') as f:
        json.dump(uniter_reverie_data,f,indent=4)

def main(args):
    reverie_path = [
        f"{args.reverie}/REVERIE_val_seen.json",
        f"{args.reverie}/REVERIE_val_unseen.json",
        f"{args.reverie}/REVERIE_test.json"]
    nav_path = [
        f"{args.nav}/submit_val_seen.json",
        f"{args.nav}/submit_val_unseen.json",
        f"{args.nav}/submit_test.json"]
    uniter_reverie = [
        "utils_grounding/uniter/REVERIE_val_seen.json",
        "utils_grounding/uniter/REVERIE_val_unseen.json",
        "utils_grounding/uniter/REVERIE_test.json"]
    if not os.path.exists(f"utils_grounding/uniter"):
                os.makedirs(f"utils_grounding/uniter")
    for i in range(3):
        nav_to_reverie(reverie_path[i], nav_path[i], uniter_reverie[i])

if __name__ == "__main__":
    tic = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverie')
    parser.add_argument('--nav')
    args = parser.parse_args()
    main(args)
    toc = time.time()
    print("time:", toc - tic)


