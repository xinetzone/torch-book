cd tests && mkdir -p tasks && cd tasks
# git clone git@github.com:xinetzone/mmpretrain.git
git clone https://github.com/xinetzone/mmpretrain.git
git clone https://github.com/xinetzone/mmdeploy.git
git clone https://github.com/xinetzone/mmengine.git
git clone https://github.com/xinetzone/mmdetection.git
git clone https://github.com/xinetzone/mmagic.git
git clone https://github.com/open-mmlab/mmcv.git
cd mmpretrain
git remote add upstream https://github.com/open-mmlab/mmpretrain
git pull upstream main
cd mmdeploy
git remote add upstream https://github.com/open-mmlab/mmdeploy
git pull upstream main
cd mmengine
git remote add upstream https://github.com/open-mmlab/mmengine
git pull upstream main
cd mmdetection
git remote add upstream https://github.com/open-mmlab/mmdetection
git pull upstream main
cd mmagic
git remote add upstream https://github.com/open-mmlab/mmagic
git pull upstream main
