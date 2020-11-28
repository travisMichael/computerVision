Python Version: 3.6.12

Libraries:
numpy
cv2
pymaxflow


External Data:
Drop the following datasets into the directory "input_images"

https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/cones-png-2.zip
https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/tsukuba.zip

File structure should look like this:
input_images/cones/im2.png

Instruction to run program:
To run all stereo algorithms, run this command:

python experiment.py

The program will output disparity maps and ground truth comparison images. The 
program will also print the error percentages to the terminal.

Disparity Maps:
cones_disparity_1_a.png
cones_disparity_1_b.png
cones_disparity_1_c.png
tsukuba_disparity_1_a.png
tsukuba_disparity_1_b.png
tsukuba_disparity_1_c.png

Ground truth comparison images:
1_a_comparision.png
1_b_comparision.png
1_c_comparision.png
1_a_comparision.png
1_b_comparision.png
1_c_comparision.png