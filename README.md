# sigmorphon2016-system
University of Helsinki contribution to the SIGMORPHON 2016 shared task on morphological reinflection

Further description can be found in the system paper submitted to the
SIGMORPHON workshop (to be held on August 11 at ACL 2016 in Berlin).
For now, a
[preliminary version of the paper is available](paper/sigmorphon.pdf).

The system is implemented in [Keras](https://keras.io/), and there are two
versions: [the shared task submission](./sigmorphon-submission.py) and
[the convolutional system](./sigmorphon-conv.py) which was finished after the
shared task deadline, but is described in the paper.

Both programs assume that the shared task data are available in the
directory `../sigmorphon2016`, which should be cloned from
[here](https://github.com/ryancotterell/sigmorphon2016).

Execute with e.g. `python3 sigmorphon-conv.py 123 german navajo`
which will write output to the `models` directory (which is assumed to exist!)
using the experiment identifier `123` and training models for German and
Navajo.

The particular experiments to run currently have to be hard-coded in
`run_experiments()` -- sorry about this, adding a convenient UI was not
a priority for the shared task.

