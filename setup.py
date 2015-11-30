from distutils.core import setup

setup(
    name='chimera_autoalign',
    version='0.0.1',
    packages=['chimera_autoalign', 'chimera_autoalign.util' , 'chimera_autoalign.controllers'],
    scripts=['scripts/chimera-autoalign'],
    url='https://github.com/tribeiro/chimera-autoalign',
    license='GPL v2',
    author='Tiago Ribeiro',
    author_email='tribeiro@ufs.br',
    description='Auto align telescope hexapod using zernike coefficient'
)
