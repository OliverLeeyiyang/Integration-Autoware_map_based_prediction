# AUTOWARE-INTEGRATION

A repository for prediction integration in EDGAR project. containing the `tum_prediction` package and the runtime evaluation results.

Recommand environment: Ubuntu 22.04 + ROS2 Humble

## Directory Structure

```
├── ...  
├── runtime evaluation results 
       ├── ...  
├── src/tum_prediction                              # tum_prediction package
       ├── config
            ├── map_and_node_params.yaml            # Yaml file for ROS2 node
       ├── launch                                   # Contains two launch files for prediction nodes
            ├── parallel_map_based_prediction.launch.xml
            ├── routing_based_prediction.launch.xml
       ├── encoder_model_weights                    # Contains weights for encoder model
            ├── ...
       ├── prediction_model_weights                 # Contains weights for prediction model
            ├── ...
       ├── norms                                    # Contains weights for normalization
            ├── ...
       ├── sample_map                               # Contains two version of test maps
            ├── DEU_GarchingCampus-1
            ├── old_DEU_GarchingCampus-1
       ├── test                                     # Contains pytest files
            ├── ...
       ├── tum_prediction                           # Contains all nodes and utilities
            ├── __init__.py
            ├── image_path_generator_pil.py
            ├── image_path_generator.py
            ├── map_loader.py
            ├── nn_model.py
            ├── node_map_based_prediction.py        # Key node for integration testing
            ├── node_methods_test.py
            ├── node_path_evaluation.py
            ├── node_routing_based_prediction.py    # Key node for NN-based prediction
            ├── node_visualization.py
            ├── original_path_generator.py
            ├── utils_interpolation.py
            ├── utils_nn_model.py
            └── utils_tier4.py
       ├── package.xml                              # ROS package and dependencies definition.
       └── setup.py                                 # setup.py for ROS package
├── build.sh                                        # sh file for building the package
├── requirements.txt                                # Pip requirements
└── Readme.md 
```

## Two options for package development

Following will be an instruction about how to set up local environment.

1. If you want to further develop the package, you could start from [Setup locally](#10-setup-locally) and you may not need [Run with docker and microservices](#20-run-with-docker-and-microservices).
2. But if you only want to test or see the result of the neural network-based trajectory prediction, you can jump to [Run with docker and microservices](#20-run-with-docker-and-microservices).

## 1.0 Setup locally

### 1.1 Install Autoware

If the environment of Planning simulation is already there, then pass this step.

[https://autowarefoundation.github.io/autoware-documentation/main/installation/](https://autowarefoundation.github.io/autoware-documentation/main/installation/)

### 1.2 Git clone the autoware-integration repository to local directory

[https://gitlab.lrz.de/prediction/autoware-integration](https://gitlab.lrz.de/prediction/autoware-integration)

### 1.3 Build package

Open a terminal, and run the following command. It's recommanded to build like this, or you can also use pip install all requirements and then use colcon build.

```bash
cd ~/autoware-integration
chmod +x build.sh
./build.sh
```

### 1.4 Run the simulation

In one terminal: run the planning simulation.

```bash
ros2 launch autoware_launch planning_simulator.launch.xml map_path:=$HOME/autoware-integration/src/tum_prediction/sample_map/DEU_GarchingCampus-1 vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit
```

In another terminal: run the routing_based_prediction_node.

```bash
ros2 launch tum_prediction routing_based_prediction.launch.xml map_path:=$HOME/autoware-integration/src/tum_prediction/sample_map/DEU_GarchingCampus-1
```

You can change the `map_path` here to your map to test another map.

## 2.0 Run with docker and microservices

When you only want to see the capability of the neural network-based prediction and don't want to develop it, a good choice is to launch them with docker. But to do so, you need to make sure your computer has at least 80GB free space!

### 2.1 Git clone the repo tum_launch to local directory

[https://gitlab.lrz.de/av2.0/tum_launch](https://gitlab.lrz.de/av2.0/tum_launch)

And choose branch `prediction_integration`.

After this, you need to clone the repository [autoware-integration repository](#12-git-clone-the-autoware-integration-repository-to-local-directory). And change the `map_path` parameter in /tum_launch/compose_launch/.env to the map in this repositories, which is:

```bash
map_path=/home/#{user}/autoware-integration/src/tum_prediction/sample_map/DEU_GarchingCampus-1
```

### 2.2 Install Docker

If the environment of Docker is already there, then pass this step.

[https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

### 2.3 Download dockers to local machine using docker compose

Run in a terminal:

```bash
sudo docker login gitlab.lrz.de:5005
Username: ge89yes
Password: [Use your Username and Password here!]
cd ~/tum_launch/compose_launch
sudo docker compose -f rviz.yml up
```

After this, open another terminal and run

```bash
cd ~/tum_launch/compose_launch
sudo docker compose --env-file .env -f tum_prediction_sim.yml up
```

#### 2.3.1 Troubleshooting for rviz

If nothing show up after you running `sudo docker compose -f rviz.yml up` ,you can:

Execute `xhost +"local:docker@"` on your host machine.
Or `startx` before run `sudo docker compose -f rviz.yml up`

### 2.4 Run the simulation by microservices

Docker files should be run automatically after step 2.3.

## 3.0 Check the prediction results in autoware planning simulation

After the step 1 or step 2, click on the RVIZ software opened by Planning Simulation.

Then find the `Add` button of the `Displays` part at the top left corner,
-> ckick it and slect `By topic`
-> scroll down that small page to the bottom
-> click `MarkerArray` in `/nn\_pred\_path\_marker` topic
-> click `OK` button

There should be a new display in "Displays" area.

Then click the `2D Pose Estimate` to add a car on the map, and click `2D Dummy Car` to add an object to be followed. Instructions about how to do this step: [https://autowarefoundation.github.io/autoware-documentation/main/tutorials/ad-hoc-simulation/planning-simulation/](https://autowarefoundation.github.io/autoware-documentation/main/tutorials/ad-hoc-simulation/planning-simulation/)

## 4.0 Further Test with another map

First, please make sure your map directory looks similar to this:

```
├── NAME_OF_MAP 
       ├── lanelet2_map.osm         # OSM map that can be loaded by UTM projector
       ├── map_config.yaml          # Contains map_origin parameters
       ├── map_projector_info.yaml  # Might not be needed for old Autoware version
       ├── ...
```

### 4.1 If you followed Step 1

1. Change the `map_path` parameter for two launch files at [Run the simulation](#14-run-the-simulation).
2. Run two launch files again.

### 4.2 If you followed Step 2

1. Change the `map_path` parameter in /tum_launch/compose_launch/.env to your map path.
2. Run dockers again.

### New and old version of autoware map

Since the map loader in Autoware is changed in the latest version(23.11.2023), one file is added to the map directory: `map_projector_info.yaml`. To use the old version map, we should modify this file ourselves, more information on website: [https://autowarefoundation.github.io/autoware.universe/main/map/map_projection_loader/](https://autowarefoundation.github.io/autoware.universe/main/map/map_projection_loader/).
