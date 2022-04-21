NUM_GPUS=1
output_dir="submit_file"
input_nav_dir="--nav input_nav_dir"
reverie_dir="--reverie Downloads/REVERIE"
boxes_dir="Downloads/BBoxes_v2"

annotation="--annotation txt_db/ann/reverie/refs.p 
                           txt_db/ann/reverie/instances.json 
                           txt_db/ann/iid2bb_id/iid_to_ann_ids.json "
inf_flag="--img_db img_db/ --output_dir weights --checkpoint best"
python utils_grounding/nav_to_reverie.py $reverie_dir $input_nav_dir
python utils_grounding/generate_p.py --boxes_dir $boxes_dir

python prepro.py $annotation --output txt_db/txt/reverie_val_seen.db
python prepro.py $annotation --output txt_db/txt/reverie_val_unseen.db
python prepro.py $annotation --output txt_db/txt/reverie_test.db

horovodrun -np ${NUM_GPUS} python inf.py $inf_flag --txt_db txt_db/txt/reverie_val_seen.db 
horovodrun -np ${NUM_GPUS} python inf.py $inf_flag --txt_db txt_db/txt/reverie_val_unseen.db 
horovodrun -np ${NUM_GPUS} python inf.py $inf_flag --txt_db txt_db/txt/reverie_test.db 

python utils_grounding/gen_submit_file.py --output_dir $output_dir
