UNET 3D plus attention using Keras/Tensorflow
--------------------------------------------------------------------------------------------------------------------------

Unet with attention gated units layer based on Vinod Github:  https://github.com/robinvvinod/unet

There is a directory where unmodified MRI Flair images  and respective masks are present, one in each directory but with the same name ie. 11.nii

Data patches are created into NewPatches folder and after that the train starts and the model is saved into models folder and prediction into prediction folders.

To run (once created patches):

    /usr/bin/python3.8 -m pip install requirements.txt
    /usr/bin/python3.8 doit.py

Also you can use the docker, that is the docker folder. To build the docker image type:

    build_docker.sh

To run the docker container, where /home/user/project is where the project data lives in your computer:

    docker run --rm -it --shm-size=1024m -v ${PWD}/3DSegmentation:/data unet-attention:v1.0 /opt/conda/bin/python3.8 /data/doit.py




![image](https://github.com/DrDiazHurtado/3DUnetSegmentationAttention/assets/100340828/8fe170de-6e80-4c50-8880-f502b434c0b2)
