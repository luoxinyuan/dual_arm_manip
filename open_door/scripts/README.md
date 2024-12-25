## Description:

This readme file introduces the calibration and trajectory recording scripts of the dual-arm manipulation project.

At the beginning
1. Connect the camera, and robot to the laptop.
2. open an anaconda prompt window (can be found by searching in Windows):
    ```bash
    cd C:\Users\arpit\Desktop\DoorBot-master\open_door\scripts
    conda activate doorbot

## Arm calibration:

1. run
    ```bash
    python run_arm_right.py

Note: this is a script that includes all functions to control the robot arm, you can try each of them to get familiar with controlling the arm.

2. input "calib" command, and the robot will move to the calibration position

Note: The robot arm will directly go to the calibration position, make sure there is no obstacle in front of the robot (eg: table).

3. Move the robot arm under the camera manually.

4. input "calibration" command, and the robot will start auto calibration.

Note: this calibration is for the transformation between the arm coordinate and the top view camera coordinate. Make sure the April tag appears in the camera and the surface of it is parallel to the table surface. The robot will only rotate the fifth joint with a step of 10 degrees to collect joint angle and April tag positions and directions, which will be saved as "robot_positions_{arm}.json" and "recorded_positions_{arm}.json" in the current directory.

Please refer to the "calibration" function in arm.py for details.

## Calibration test:

All calibration functions locate at "calibrate_from_json.py".
Make sure you have the correct:
1. `recorded_positions_right.json`
2. `robot_positions_right.json`
3. `curve_points.json`

Note: the rz in 1 should be in an increasing sequence. If the April tag rotation is close to the limitation (eg: ’rz1=-175‘, ’rz2=175‘), please rotate April tag or add an offset.

Run the following command to use 1 and 2 to calibrate and transfer 3 to the coordinate in the robot frame and save as "base_points_transformed.json":
    
    python calibrate_from_json.py


## Trajectory test:

You can test the accuracy and reachability of this transferred trajectory: ’base_points_transformed.json‘.
1. run
    ```bash
    python run_arm_right.py
   
3. input command "traj"

## Record trajectory:

run 
    ```bash
    python collect_trajectory.py

follow the instructions to record the trajectory.

The script will record both left and right trajectory and direction in camera coordinates, the format is [x, y, 0, 0, 0, rz], which are the x and y pixels and rz is the rotation of the April tag. The raw video of the whole process will be recorded at the same time. All those data will be saved in the directory "dmp_traj".

If you have other questions, you can chat me on Slack, through email: xl153@illinois.edu, or WeChat: lxynzmzmc
Xinyuan Luo
