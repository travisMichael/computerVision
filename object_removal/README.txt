This project implements the concepts described in the following paper.

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf

Instructions for running code

The following python libraries were used: numpy, cv2, and sys.

The environment for running the code uses the same anaconda environment for the previous assignments in this class

If you do not have the environment already created, then run the following command:

conda env create -f cs6475.yml

Then activate the environment by running the following command:

source activate CS6475

Make sure to create the following directories at the root project level:
mkdir image_set_1
mkdir image_set_2
mkdir image_set_3

Then download the images with 'original' and 'marker' in their names from the following hosted directories. Put the
downloaded images in the corresponding folders from the previous step.

Image set 1:
https://github.com/travisMichael/computerVision/tree/master/image_set_1

Image set 2:
https://github.com/travisMichael/computerVision/tree/master/image_set_2

Image set 3:
https://github.com/travisMichael/computerVision/tree/master/image_set_3

To run the algorithm on image set 1, run the following command:

python final_project.py island

To run the algorithm on image set 2, run the following command:

python final_project.py shower

To run the algorithm on image set 3, run the following command:

python final_project.py anne

The output files for each image set can be found under their associated directory names.
