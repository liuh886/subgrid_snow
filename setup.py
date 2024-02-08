from setuptools import setup, find_packages

setup(
    name='xsnow',  # Replace with your package name
    version='0.1.0',  # Replace with your package version
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        # List your project's dependencies here
        # e.g., 'numpy>=1.18.5',
        # 'pandas>=1.0.5',
    ],
    # Additional metadata about your package
    author='Zhihao',
    author_email='liuzhihao109@foxmail.com',
    description='A tool box for DEM coregistration and snow depth regression',
    url='https://github.com/liuh886/subgrid_snow',  # Project home page or repository URL
    # List additional classifiers (https://pypi.org/classifiers/)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
