cd gaussian_splatting
SCENE="bicycle"

# To avoid running out of memory, increase the densification threshold and interval, and decrease iter limit
python train.py -s ../dataset/360_v2/$SCENE -m output/$SCENE --densify_grad_threshold 0.0002 \
    --densification_interval 100 --densify_until_iter 15000

mv output/$SCENE ../scene_models/
cd../