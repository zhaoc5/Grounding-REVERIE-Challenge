import json
import entity
import os

def dict_to_entities(filename):
    basename = os.path.basename(filename)
    index_first_underline = basename.index("_") + 1
    index_second_underline = basename.index(".")
    split = basename[index_first_underline:index_second_underline]
    train_entity_dict = {}
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        for key in load_dict:
            train_entity_ = entity.train_entity(key)
            train_entity_.set_split(split)
            train_entity_dict[train_entity_.id] = train_entity_
    return train_entity_dict

def load_train_json(filename_list):
    entity_dict = {}
    for filename in filename_list:
        train_entity_dict = dict_to_entities(filename)
        print(f"cur: {len(train_entity_dict)}")
        entity_dict.update(train_entity_dict)
        print(f"sum: {len(entity_dict)}")
    return entity_dict

def load_bb_json(filename):
    scene_id = os.path.basename(filename).split("_")[0]
    viewpoint_id = os.path.basename(filename).split("_")[1]
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        bb_entity_ = entity.bb_entity(scene_id, viewpoint_id, load_dict)
    return bb_entity_
