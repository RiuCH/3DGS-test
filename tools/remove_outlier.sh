# SCENE="bicycle"
# SCENE_PATH=scene_models/$SCENE/point_cloud/iteration_30000
# 3dgsconverter -i $SCENE_PATH/point_cloud.ply -o $SCENE_PATH/point_cloud_clean.ply -f cc --density_filter --remove_flyers
# 3dgsconverter -i $SCENE_PATH/point_cloud_clean.ply -o $SCENE_PATH/point_cloud_clean.ply -f 3dgs 

SCENE="hotdog"
SCENE_PATH=object_models/$SCENE/point_cloud/iteration_30000
3dgsconverter -i $SCENE_PATH/point_cloud.ply -o $SCENE_PATH/point_cloud_clean.ply -f cc --density_filter --remove_flyers
3dgsconverter -i $SCENE_PATH/point_cloud_clean.ply -o $SCENE_PATH/point_cloud_clean.ply -f 3dgs 