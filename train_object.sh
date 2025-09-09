cd gaussian_splatting
OBJECT="hotdog"
python train.py -s ../dataset/nerf_synthetic/$OBJECT -m output/$OBJECT
mv output/$OBJECT ../object_models/
cd ../