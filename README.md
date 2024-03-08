# My adapted implementation of SymmetryNet

## About

This is an implementation of an adaptation of SymmetryNet that focuses on
the prediction of planar reflective symmetries. It doesnt use the dense prediction scheme
described in the original work.

The final implementation used is the one of the center_n_normals_net. It can be used
using the torch lighting cli interface. The simple_net also works.

The implementation contained in the symmetry_net file is currently broken due the implementation
of the loss function. The results presented in my final work are from a center_n_normals_net that predicted
the test dataset and was evaluated using the file evaluation.py from scripts. I will upload the model that obtained the
last result later.
 



