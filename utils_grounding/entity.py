import json

class train_entity(object):
    def __init__(self, dict):
        self.dict = dict
        self.parse_dict(dict)

    def parse_dict(self, dict):
        self.scan = dict['scan']
        self.id = dict['id']
        self.path = dict['path']
        self.end_viewpoint = self.path[-1]
        self.heading = dict['heading']
        self.instructions = dict['instructions']

    def set_split(self, split):
        self.split = split

    def to_string(self):
        print("scan", self.scan)
        print("id", self.id)
        print("path", self.path)
        print("end_viewpoint", self.end_viewpoint)
        print("heading", self.heading)
        print("instructions", self.instructions)

class bb_view_obj_entity(object):
    def __init__(self, scene_id, viewpoint, view_id, objId, dict):
        self.scene_id = scene_id
        self.viewpoint = viewpoint
        self.view_id = view_id
        self.objId = int(objId)
        self.dict = dict
        self.parse_dict(dict)

    def parse_dict(self, dict):
        self.name = dict['name']
        self.category = dict['category']
        self.bbox2d = dict['bbox2d']
        self.pixels = dict['pixels']

    def to_string(self):
        return f"'name': {self.name}, 'category': {self.category}, 'bbox2d': {self.bbox2d}, 'pixels': {self.pixels} , 'objId': {self.objId}"

class bb_view_entity(object):
    def __init__(self, scene_id, viewpoint,  view_id, dict):
        self.scene_id = scene_id
        self.viewpoint = viewpoint
        self.view_id = view_id
        self.dict = dict
        self.parse_dict(dict)

    def parse_dict(self, dict):
        self.bb_view_obj_entity_arr = []
        for objId in dict:
            bb_view_obj_entity_ = bb_view_obj_entity(self.scene_id, self.viewpoint, self.view_id, objId, dict[objId])
            self.bb_view_obj_entity_arr.append(bb_view_obj_entity_)
        return self.bb_view_obj_entity_arr

    def to_string(self):
        to_string_value = f"'scene_id': {self.scene_id}, 'view_id': {self.view_id}\n"
        for item in self.bb_view_obj_entity_arr:
            to_string_value += "        " + item.to_string() + "\n"
        return to_string_value

class bb_entity(object):
    def __init__(self, scene_id, viewpoint, dict):
        self.scene_id = scene_id
        self.viewpoint = viewpoint
        self.dict = dict
        self.parse_dict(dict)

    def parse_dict(self, dict):
        self.bb_view_entity_dict = {}
        for view_id in dict:
            bb_view_entity_ = bb_view_entity(self.scene_id, self.viewpoint, view_id, dict[view_id])
            self.bb_view_entity_dict[view_id] = bb_view_entity_
        return self.bb_view_entity_dict

    def get_all_bb(self):
        all_bb = []
        for key in self.bb_view_entity_dict:
            for bb_entity in self.bb_view_entity_dict[key].bb_view_obj_entity_arr:
                all_bb.append(bb_entity)
        return all_bb

    def get_viewId_by_objId(self, objId, filter_ix_num=36):
        view_id_arr = []
        view_pixel_arr = []
        for key in self.bb_view_entity_dict:
            for bb_entity in self.bb_view_entity_dict[key].bb_view_obj_entity_arr:
                if bb_entity.objId == objId:
                    view_id_arr.append(bb_entity.view_id)
                    view_pixel_arr.append(bb_entity.pixels)
                    break
        return view_id_arr, view_pixel_arr

    def get_bb_by_viewId_objId(self, viewId, objId):
        for key in self.bb_view_entity_dict:
            if self.bb_view_entity_dict[key].view_id == viewId:
                for bb_entity in self.bb_view_entity_dict[key].bb_view_obj_entity_arr:
                    if bb_entity.objId == objId:
                        return bb_entity

    def get_bbIds_by_viewId(self, viewId):
        bbIds = {}
        for bb in self.bb_view_entity_dict[viewId].bb_viewpoint_obj_entity_arr:
            bbIds[bb.objId] = bb
        return bbIds

    def to_string(self):
        to_string_value = f"'scene_id': {self.scene_id} \n"
        for key in self.bb_view_entity_dict:
            to_string_value += "    " + \
                self.bb_view_entity_dict[key].to_string() + "\n"
        return to_string_value

class image_entity(object):
    def __init__(self, dict):
        self.parse_dict(dict)

    def parse_dict(self, dict):
        self.id = dict["id"]
        self.ann_ids = dict["ann_ids"]
        self.filename = dict["filename"]
        self.height = dict["height"]
        self.width = dict["width"]

    def to_string(self):
        return f"image entity id:{self.id} "

class image_entity_arr(object):
    def __init__(self, image_json_filename):
        self.entity_dict = {}
        with open(image_json_filename, 'r') as load_f:
            load_dict = json.load(load_f)
            self.parse_dict(load_dict)

    def parse_dict(self, dict):
        for key in dict:
            self.entity_dict[key] = image_entity(dict[key])

    def get_image_entity_by_image_name(self, image_name):
        for key in self.entity_dict:
            if self.entity_dict[key].filename.split(".")[0] == image_name:
                return self.entity_dict[key]
