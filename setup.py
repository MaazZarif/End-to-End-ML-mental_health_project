from setuptools import find_packages,setup
from typing import List



HYPHEN_E_DOT = "-e ."

def get_requirements(filepath:str)->List[str]:
    '''
    this will return the list of requirements.'''

    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace('\n',"") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)


    return requirements




setup(
    name="mental_health_project",
    version="0.0.1",
    author="maazzarif",
    author_email="maazzarif04@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')
)