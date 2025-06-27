# 📊 Collect Your Own Human Demonstration

## Requirements
- We uesd 4 Realsense D455 cameras for collecting the human demonstration. The codebase supports human demonstration collecting as long as you have 2 or more cameras.

- On the machine(either DROID laptop or workstation) you would like to process the data, install [HaMer](https://github.com/RUreadyo/hamer), following the instruction. Make sure to use the cloned submodule, as we have some changes for DemoDiffusion.

## Camera Calibration 
With the DROID robot setup, follow the instructions in [Neural MP](https://github.com/mihdalal/neuralmotionplanner) for calibrating the camera. Once you are done, copy your updated camera yaml file to     

- $PATH_TO_DEMODIFFUSION/demodiffusion/collect/camera_calibration/camera.yaml in the machine you installed HaMer.

- $PATH_TO_DEMODIFFUSION/demodiffusion/deploy/manimo/manimo/conf/camera/multi_real_sense_calibration.yaml in DROID laptop.

        
## Collecting Human Demonstration
In DROID laptop, 
1. Update the $PATH_TO_DEMODIFFUSION/demodiffusion/deploy/manimo/manimo/conf/sensors.yaml to use multi_real_sense_calibration instead of multi_real_sense_pizero.
2. Then, run 
    ```   
    conda activate manimo-latest
    
    cd PATH_TO_DEMODIFFUSION/demodiffusion/deploy/manimo/manimo/scripts

    python human_loop.py # before running this, change the storage_path's task name to yours (default: closelaptop).
    ```

    This would start collecting the human demonstration. Once done, press 'q' to finish recording.
3. After, you would see the collected demonstration as .pkl file in the storage_path you specified. Send this data to the machine you installed HaMer.


## Preprocessing Human Demonstration
1. In the machine you installed HaMer, run
    ```   
    conda activate hamer

    cd PATH_TO_DEMODIFFUSION/demodiffusion/collect/camera_calibration

    python cam.py
    ```
    This would save the camera intrinsics & extrinsics as numpy.

2. Then, run
    ```   
    conda activate hamer

    cd PATH_TO_DEMODIFFUSION/demodiffusion/collect/hammer

    python preprocess.py --task_name $TASK_NAME --traj_num $TRAJ_NUM 
    ```

    If you have 2 or 3 cameras, tune the reprojection threshold (in traingulate_with_ransac function) to be higher.

    Once done, it would save the human 3D keypoints under human_demos/$TASK_NAME/$TRAJ_NUM/processed_3d. 

3. To visualize HaMer reconstruction as videos, run 
    ```   
    conda activate hamer

    cd PATH_TO_DEMODIFFUSION/demodiffusion/collect/hammer

    python save_video_processed.py #update the task and traj num
    ```

    This would save the videos of HaMer results in human_demos/$TASK_NAME/$TRAJ_NUM.


