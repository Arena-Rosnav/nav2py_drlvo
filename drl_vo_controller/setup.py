from setuptools import find_packages, setup

package_name = 'drl_vo_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kien',
    maintainer_email='tonmanhkien@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller=drl_vo_controller.drl_vo_controller:main',
            'test_logger=drl_vo_controller.test_logger:main'
        ],
    },
)
