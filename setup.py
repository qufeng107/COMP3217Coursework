from setuptools import setup, find_packages

setup(
    name="Detection of Manipulated Pricing in Smart Energy CPS Scheduling",
    version="1.0",
    author="Feng Qu",
    author_email="qufeng107@qq.com",
    description="Detection of Manipulated Pricing. Calculate scheduling results",
    url="https://github.com/qufeng107/COMP3217Coursework.git", 
    packages=find_packages(),
    install_requires=['keras~=2.8.0','matplotlib~=3.5.1','numpy~=1.22.3','openpyxl~=3.0.10','pandas~=1.4.2','PuLP~=2.6.0','scikit_learn~=1.1.1','setuptools~=61.3.1','tensorflow~=2.8.0']
)