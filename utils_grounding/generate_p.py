import time
import json
import pickle
import utils
import os
from pytorch_pretrained_bert import BertTokenizer
import argparse
from tqdm import tqdm

def generate_p_file(json_filename_list, boxes_dir, output_refs_filename, output_refs_json_filename, output_instances_filename, output_iid_to_ann_ids_filename, prefix, open_check_flag, filter_ix_num = 36):
    refs_dict = []
    instances_dict = {}
    instances_images_dict = {}
    instances_annotations_dict = {}
    instances_categories_arr = []
    iid_to_ann_ids_dict = {}
    iid_to_ann_ids_dict_sub = {}
    image_ordered_id = 0
    image_unique_id = 0
    anno_ordered_id = 0
    anno_unique_id = 0
    refs_unique_id = 0
    refs_sent_id = 0
    anno_unique_id_arr = []
    train_entity_dict = utils.load_train_json(json_filename_list)
    toker = 'bert-base-cased'
    toker = BertTokenizer.from_pretrained(toker, do_lower_case='uncased' in toker)
    if open_check_flag:
        to_del_ids = []
        to_del_filenames = []
    for id in tqdm(train_entity_dict): 
        if open_check_flag:
            if id not in to_del_ids:
                to_del_ids.append(id)
        bb_filename = train_entity_dict[id].scan + "_" + train_entity_dict[id].end_viewpoint + ".json"
        if open_check_flag:
            if bb_filename not in to_del_filenames:
                to_del_filenames.append(bb_filename)
        bb_filename = os.path.join(boxes_dir, bb_filename)
        bb_entity_ = utils.load_bb_json(bb_filename)
        instruction_len = len(train_entity_dict[id].instructions)
        for view_id in range(36):
            view_id = str(view_id)
            image_id = train_entity_dict[id].scan + "_" + train_entity_dict[id].end_viewpoint + "_" + view_id.rjust(2,"0")
            filename = image_id + ".jpg"
            h5name = f'{image_id}.h5'
            height = 480
            width = 640
            if filename not in instances_images_dict:
                image_unique_id = image_ordered_id
                instances_images_dict[filename] = {
                    "id": image_unique_id,
                    "file_name": filename,
                    "height": height,
                    "width": width,
                }
            else:
                image_unique_id = instances_images_dict[filename]["id"]
                image_ordered_id -= 1
            view_entity_arr = bb_entity_.bb_view_entity_dict[view_id]
            check=0
            for bb in view_entity_arr.bb_view_obj_entity_arr:
                character_id = train_entity_dict[id].scan + "_" + train_entity_dict[id].end_viewpoint + "_" + view_id + "_" + str(bb.objId)
                if character_id not in instances_annotations_dict:
                    anno_unique_id = anno_ordered_id
                    instances_annotations_dict[character_id] = {
                        "id": anno_unique_id,
                        "character_id": character_id,
                        "area": bb.pixels,
                        "bbox": bb.bbox2d,
                        "image_id": image_unique_id,
                        "category_id": bb.category,
                        "iscrowd": 0
                    }
                else:
                    anno_unique_id = instances_annotations_dict[character_id]["id"]
                    anno_ordered_id -= 1
                if bb.category not in [instance["id"] for instance in instances_categories_arr]:
                    instances_categories_arr.append({
                        "id": bb.category,
                        "supercategory": bb.name,
                        "name": bb.name
                    })
                sent_ids = list(range(refs_sent_id, refs_sent_id+instruction_len))
                if check == 0:
                    check+=1
                    refs_sent_id += instruction_len
                    instructions_list = []
                    for instr_index in range(len(train_entity_dict[id].instructions)):
                        instructions_list.append({
                                'sent_id': sent_ids[instr_index],
                                'sent': train_entity_dict[id].instructions[instr_index]
                            }
                        )
                    height = 480
                    width = 640
                    refs_dict.append({
                        "id": id,
                        "split": train_entity_dict[id].split,
                        "image_id": image_unique_id,
                        "h5_name": h5name,
                        "ref_id": refs_unique_id,
                        "sentences": instructions_list,
                        "file_name": filename,
                        "sent_ids": sent_ids
                    })
                    refs_unique_id += 1
                anno_unique_id_arr.append(anno_unique_id)
                anno_ordered_id += 1
            iid_to_ann_ids_dict_sub[image_unique_id] = anno_unique_id_arr
            anno_unique_id_arr = []
            image_ordered_id += 1
    instances_dict["images"] = list(instances_images_dict.values())
    instances_dict["annotations"] = list(instances_annotations_dict.values())
    instances_dict["categories"] = instances_categories_arr
    iid_to_ann_ids_dict['iid_to_ann_ids'] = iid_to_ann_ids_dict_sub
    with open(output_refs_filename, "wb") as f:
        pickle.dump(refs_dict, f)
        print(output_refs_filename, "loading finished ...")
    with open(output_refs_json_filename, "w") as f:
        json.dump(refs_dict, f, sort_keys=True,
                  indent=4, separators=(',', ':'))
        print(output_refs_json_filename, "loading finished ...")
    with open(output_instances_filename, "w") as f:
        json.dump(instances_dict, f, sort_keys=True,
                  indent=4, separators=(',', ':'))
        print(output_instances_filename, "loading finished ...")
    with open(output_iid_to_ann_ids_filename, "w") as f:
        json.dump(iid_to_ann_ids_dict, f, sort_keys=True,
                  indent=4, separators=(',', ':'))
        print(output_iid_to_ann_ids_filename, "loading finished ...")

def main(args):
    prefix = "reverie"
    json_filename_list = [
        "utils_grounding/uniter/REVERIE_val_seen.json",
        "utils_grounding/uniter/REVERIE_val_unseen.json",
        "utils_grounding/uniter/REVERIE_test.json"]
    if not os.path.exists("txt_db/ann/reverie"):
        os.makedirs("txt_db/ann/reverie")
    if not os.path.exists("txt_db/ann/iid2bb_id"):
        os.makedirs("txt_db/ann/iid2bb_id")
    output_refs_filename = "txt_db/ann/reverie/refs.p"
    output_refs_json_filename ="txt_db/ann/reverie/refs.json"
    output_instances_filename = "txt_db/ann/reverie/instances.json"
    output_iid_to_ann_ids_filename = "txt_db/ann/iid2bb_id/iid_to_ann_ids.json"
    open_check_flag = True
    generate_p_file(json_filename_list, args.boxes_dir, output_refs_filename, output_refs_json_filename, output_instances_filename, output_iid_to_ann_ids_filename,prefix,open_check_flag)

if __name__ == "__main__":
    tic = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxes_dir')
    args = parser.parse_args()
    main(args)
    toc = time.time()
    print("time:", toc - tic)
