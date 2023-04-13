from setuptools import setup

setup(
        name='segway.synapse.area',
        version='0.01',
        url='https://github.com/htem/segway.synapse.area',
        author='Tri Nguyen',
        author_email='tri_nguyen@hms.harvard.edu',
        license='MIT',
        packages=[
            'segway.synapse.area'
        ],
        install_requires=[
            "funlib.math",
            "funlib.geometry",
            "skimage",
            "jsmin",
            "daisy",
            "opencv-python",
        ]
)
