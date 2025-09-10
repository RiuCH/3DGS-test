conda create -n "3DGS" python=3.9 ipython
conda activate 3DGS
conda install -c nvidia cuda-toolkit=12.4
conda install -c conda-forge plyfile
conda install tqdm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
pip install joblib

cd gaussian_splatting  
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
cd ../