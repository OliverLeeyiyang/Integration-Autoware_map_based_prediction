import os

from setuptools import setup

package_name = "tum_prediction"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), ["config/map_and_node_params.yaml"]),
        (
            os.path.join("share", package_name, "launch"),
            ["launch/parallel_map_based_prediction.launch.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            ["launch/routing_based_prediction.launch.xml"],
        ),
        (
            os.path.join("share", package_name, "sample_map"),
            ["sample_map/DEU_GarchingCampus-1/lanelet2_map.osm", "sample_map/DEU_GarchingCampus-1/map_config.yaml"],
        ),
        (
            os.path.join("share", package_name, "old_sample_map"),
            ["sample_map/old_DEU_GarchingCampus-1/lanelet2_map.osm", "sample_map/old_DEU_GarchingCampus-1/map_config.yaml"],
        ),
        (
            os.path.join("share", package_name, "norms"),
            ["norms/egoVelocityNorm.pt","norms/socialInformationOfCarsInViewNorm.pt", "norms/relGroundTruthNorm.pt"],
        ),
        (
            os.path.join("share", package_name, "encoder_model_weights"),
            ["encoder_model_weights/CNN_200.pth"],
        ),
        (
            os.path.join("share", package_name, "prediction_model_weights"),
            ["prediction_model_weights/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.442764_lr0.0003_2023_08_16_19:33:00.pth"],
        ),
    ],
    zip_safe=True,
    maintainer="oliver",
    maintainer_email="ge89yes@tum.de",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "parallel_map_based_prediction_node = tum_prediction.node_map_based_prediction:main",
            "routing_based_prediction_node = tum_prediction.node_routing_based_prediction:main",
            "visual_node = tum_prediction.node_visualization:main",
            "path_evaluation = tum_prediction.node_path_evaluation:main",
            "pp_test = tum_prediction.node_methods_test:main",
        ],
    },
)
