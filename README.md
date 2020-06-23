# UBICOMP ASSIGNMENT 2 -- ACTIVITY RECOGNITION #

For this project to work, there are a few things that need to be in place.

Firstly, place into the root directory a directory called `allData`. In this folder, there should only be other folders. In each of these sub-directories, please ensure that every file either ends in `accel.txt`, `gyro.txt`, or `pressure.txt`. **Please also ensure that for each file ending in `accel.txt` there is a file ending in `pressure.txt` that otherwise has the same filename for the same time period.** For instance, if there is an `xyz-accel.txt`, there must also be an `xyz-pressure.txt`. Each sub-directory must also include a file `label.txt`.

If it is necessary to keep all the files regardless of their compatibility to the system, mark the subdirectories with the inconsistencies with a suffix '-nl' at the end of the directory. In my directory, I have appended an '-nl' to these directories of the dataset (most were missing `label.txt`):
* `akroy-nl`
* `arajasekaran-nl`
* `njaiman-nl`
* `qifanyang-nl`

Lastly, install these libraries:
* `numpy`
* `sklearn`
* `matplotlib`
* `scipy`