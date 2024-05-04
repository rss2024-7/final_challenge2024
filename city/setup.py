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
        ('share/' + package_name + '/launch', ['launch/sim_drive.launch.xml']),
        ('share/path_planning/maps', glob.glob(os.path.join('maps', '*'))),
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
        ],
    },
)
