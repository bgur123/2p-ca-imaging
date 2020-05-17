 # pyCaImaging
> Calcium imaging analysis software for Silies Lab.

## Installation
Recommended way to install and use the software is through a python virtual environment. That way, you will not have any dependency/incompatibility issues with other python packages that were installed system-wide. It allows to test and use the software with minimal possible number of packages in an isolated environment. The developers and users can use almost exactly the same configuration and set of packages across different systems which eliminates possible incompatibilities in the development process.

Here, `miniconda` distribution for Python (stripped down version of Anaconda) will be used. However, you can also use the canonical Python and create a virtual environment - this is not covered here. As long as you stick with `miniconda`, you can directly use the `environment.yml` file from `pyCaImaging` to create exactly the same virtual environment that was used for testing and development, which assures great compatibility across systems (more on that later). Otherwise, you might have to resolve dependencies on your own.

You first need to install Python 2.7 version of `miniconda` (available for MacOS X, Linux and Windows). Instructions are shown in the following link:
[Miniconda Installation](https://conda.io/miniconda.html)

If you want to know more about conda and virtual environments:
[Conda user guide](https://conda.io/docs/user-guide/)

### OS X & Linux
#### Initial configuration
The lines below assume that `conda` is not in your `PATH` environment. If it is, you can replace `miniconda2/bin/conda` in the lines with `conda`, and `miniconda2/bin/activate` with `activate`. In short, if `conda` is in `PATH`, you can directly enter the command, otherwise you need to enter the path to the binary file (`conda` and `activate` in this case).
#### Actual installation
+ Assuming you are in the `HOME` folder and `conda` is not in `PATH`, execute the following:
```sh
miniconda2/bin/conda env create -f path/to/environment.yml -n pyCaImaging
```
`environment.yml` file is in `pyCaImaging`, replace `path/to/environment.yml` with the actual path you have.
+ After you execute the line above, a `python 2.7` virtual environment named pyCaImaging has been created, with all the dependencies installed. Activate the environment using,
```sh
source miniconda2/bin/activate pyCaImaging
```
In the terminal, you should see that the environment name appears before the username, if the environment is activated.

+ Download the modified version of `sima`, and build it from source. Do *not* use the one in [PyPI-Python Package Index](https://pypi.python.org/pypi/sima/) since the source code in our repo is modified to be compatible with `pyCaImaging`. After downloading the modified `sima`, in a terminal, `cd` to the directory which contains `setup.py`
Then run,
```sh
python setup.py build
python setup.py install
```
`sima` installation is completed.
+ Next, install `roibuddy` for ROI selection;
```sh
pip install roibuddy
```
**Note:** Do not use the `--user` option in `pip` with the above command, in the current case we have activated the virtual environment and it will automatically install the package in there. If you use `--user`, then it will be installed somewhere else.
+ That's it! You can check if both `sima` and `roibuddy` are installed correctly by;
```sh
miniconda2/bin/conda list
```
#### Editor configuration
Open your favourite editor/terminal, change the present working directory to the root directory of `pyCaImaging` folder (or add it in your `PYTHONPATH` in the editor, so python can know about the functions in the package when they are called).
Once everything is done, quit the editor and deactivate the environment,
```sh
source deactivate
```

**Note:** `spyder` editor sould be already installed if you have used the `environment.yml` file during installation. For `spyder` users, setting the present working directory/`PYTHONPATH` can be done via:
> Tools -> Preferences -> Run -> General settings -> Default working directory is: the script directory

OR

> Tools -> PYTHONPATH manager -> Add the path/to/pyCaImaging

### Windows
Since OS X & Linux are both UNIX-like, command line syntax is very similar, but for Windows I need to check it. However, adjustements will be minor and can be done very quickly. If you are willing to put a guide for windows, that would be perfect. Otherwise, I will do it whenever I have time.

## Usage
Activate the environment,
```sh
source miniconda2/bin/activate pyCaImaging
```
Then you can start using the functions in `pyCaImaging`.
You can find example workflows [run-Corr.py](run-Corr.py) and [run-NoCorr.py](run-NoCorr.py) and check the help documentation for every function (not so complete yet).

## Versioning
Use [Semantic versioning](https://semver.org/) for versioning. Basically,
> Given a version number MAJOR.MINOR.PATCH, increment the:
>    1. MAJOR version when you make incompatible API changes,
>    2. MINOR version when you add functionality in a backwards-compatible manner, and
>    3. PATCH version when you make backwards-compatible bug fixes.

## Release History
+ **v1.0.00**
  - First stable release

## Authors
+ Cagatay Aydin

## Contributing
1. Create your bug fix/feature branch (`git checkout -b bugFix`)
2. Commit your changes (`git commit -am 'Description of changes'`)
3. Push to the branch (`git push origin bugFix`)
4. Create a new Pull Request

## License
This project is licensed under the GNU General Public License, version 2 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
This python software is built mainly on top of `SIMA`. If you are curious about `sima` package see:
* [Losonczy Lab](http://www.losonczylab.org/sima)
* [SIMA on GitHub](https://github.com/losonczylab/sima)

And its paper:
* Kaifosh P, Zaremba J, Danielson N, and Losonczy A. SIMA: Python software for analysis of dynamic fluorescence imaging data. Frontiers in Neuroinformatics. 2014 Aug 27; 8:77. doi: 10.3389/fninf.2014.00077.

