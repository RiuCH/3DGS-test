cd gaussian_splatting
SCENE="bicycle"
python train.py -s ../dataset/360_v2/$SCENE -m output/$SCENE
mv output/$SCENE ../scene_models/
cd../