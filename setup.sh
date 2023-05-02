pip install -r requirements.txt
mkdir data
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o data/sam_vit_h_4b8939.pth
git submodule init
git submodule update