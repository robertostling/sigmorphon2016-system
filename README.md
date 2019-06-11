# sigmorphon2016-system
University of Helsinki contribution to the SIGMORPHON 2016 shared task on morphological reinflection

A description can be found in the
[system paper from SIGMORPHON](http://aclweb.org/anthology/W/W16/W16-2003.pdf)
([BibTeX](http://www.robos.org/sections/research/robert_bib.html#Ostling2016sigmorphon)).

The system is implemented in [Keras](https://keras.io/), and there are two
versions: [the shared task submission](./sigmorphon-submission.py) and
[the convolutional system](./sigmorphon-conv.py) which was finished after the
shared task deadline, but is described in the paper.

Both programs assume that the shared task data are available in the
directory `../sigmorphon2016`, which should be cloned from
[here](https://github.com/ryancotterell/sigmorphon2016).

It is recommended to use [virtualenv](https://virtualenv.pypa.io) to set up the
dependencies of this project. To do so, execute the following commands:

```sh
python3 -m venv /path/to/virtual/environment
source /path/to/virtual/environment/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Evaluation can be executed with e.g. `python3 sigmorphon-conv.py 123 german navajo`
which will write output to the `models` directory (which is assumed to exist!)
using the experiment identifier `123` and training models for German and
Navajo. After you finished testing, execute `deactivate` to exit the virtual environment.

The particular experiments to run currently have to be hard-coded in
`run_experiments()` -- sorry about this, adding a convenient UI was not
a priority for the shared task.

