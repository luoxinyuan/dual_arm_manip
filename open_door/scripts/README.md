## Description:

This readme file introduces the calibration and trajectory recording sripts of the dual-arm manipulation project.

At the begining
1. Connect the camera, robot to the laptop.
2. open an annaconda prompt window (can be found by searching in windows):
    ```bash
    cd C:\Users\arpit\Desktop\DoorBot-master\open_door\scripts
    conda activate doorbot

## Arm calibration:
Step1: run python run_arm_right.py
Note: this is a script includes all functions to control robot arm, you can try each of them to get familiar with controlling the arm.
Step2: input "calib" command, the robot will move to calibration position
(Note: The robot arm will directly go to the calibration position, make sure there is no obstacle in front of the robot (eg: table).)
Step3: Move the robot arm under the camera manually.
Step4: input "calibration" command, the robot will start auto calibration.
(Note: this calibration is for the transformation between arm coordinate and top view camera coordinate. Make sure the april tag appears in the camera and the surface of it is parallel to the table surface. The robot will only rotate the fifth joint with a step of 10 degree to collect joint angle and apriltag positions and directions, which will be saved as "robot_positions_{arm}.json" and "recorded_positions_{arm}.json" in current directory.)

Please refer to "calibration" function in arm.py for details.

## Calibration test:
All calibration functions locate at "calibrate_from_json.py".
Make sure you have correct 1."recorded_positions_right.json", 2."robot_positions_right.json" and 3."curve_points.json"
(Note: the rz in 1 should be in a increasing sequence. If the apriltag rotation is close to the limitation (eg: rz1=-175, rz2=175), please rotate apriltag or add an offset.)
Run python calibrate_from_json.py to use 1 and 2 to calibrate and transfer 3 to coordinate in robot frame and save as "base_points_transformed.json".

## Trajectory test:
You can test the accuracy and reachability of this transfered trajectory: "base_points_transformed.json".
Step1: run python run_arm_right.py
Step2: input command "traj"

## Record trajectory:
run python collect_trajectory.py
follow the instruction to record trajectory.
The script will record both left and right trajectory and direction in camera coordinate, the format is [x, y, 0, 0, 0, rz], which are the x and y pixel and rz is the rotation of theapril tag. The raw video of the whole process will be recorded at the same time. All those data will be saved in directory "dmp_traj".

If you have other questions, you can chat me on slack, through email: xl153@illinois.edu or wechat: lxynzmzmc
Xinyuan Luo