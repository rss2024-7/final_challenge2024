from setuptools import setup
import glob
import os

package_name = 'city'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/drive.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/drive_staff.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/drive_path_plan.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/sim_drive.launch.xml']),
        ('share/' + package_name + '/launch', ['launch/sim_follow.launch.xml']),
        ('share/city/maps', glob.glob(os.path.join('maps', '*'))),
        ('share/city/example_trajectories', glob.glob(os.path.join('example_trajectories', '*.traj'))),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='michaelszeng@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_follower = city.trajectory_follower:main',
            'trajectory_planner = city.trajectory_planner:main',
            'stop_detector = city.stop_detector:main',
            'light_detection = city.light_detection:main',
            'pedestrian_avoider = city.pedestrian_avoider:main',
        ],
    },
)
