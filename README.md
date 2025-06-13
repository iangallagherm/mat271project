# Problem

Have fixed set of wave emitters at surface of domain that provide a source signal. The acoustic wave equation

$\nabla^2 p(x,y,t) - \frac{1}{v(x,y)} \frac{\partial p(x,y,t)}{\partial t^2} = s(x,y,t)$

is solved for a given velocity map $v(x, y)$. Receivers at the surface provide a spatio-temporal recording of the amplitude information of the generated seismic pressure wave $p(x_i, t)$ at a discrete set of surface locations $x_1, \ldots x_n$.

From this, we would like to perform full waveform inversion in order to recover the underlying velocity map. This is a highly nonlinear nonconvex problem.

# Approaches

## Forward Problem

Solve in frequency domain

Recursive linearization - Carloz Brouges

High resolution inverse scattering in two dimensions via …

Optimal transport for full waveform inversion Bjorn, Yunan Yang

FD - Not all frequencies properly resolved. cycle skipping problems, etc. 

## Pure ML

Upon reviewing the model implementations in the Kaggle project, the highest scoring public models employ a similar overall architecture:

- Use of ConvNext: A next generation CNN architecture that is competitive with vision transformers.
- Use of a pretrained open weight model to initialize the encoder backbone.
- Use of the Adam optimizer which is a variation of stochastic gradient descent methods
- Cosine annealing time step
- Various modifications to the training data set including:
    - Resampling (And up/downsampling) to reshape the input data or decrease the size at the expense of some loss of information
    - Using horizontal flips of training data to learn features that are invariant upon reflections.

The public models do not seem to incorporate the physics into the models at all. This could be because the physics is not improving the accuracy or because everyone on Kaggle is a wannabe data scientist.

## Packages

A list of available software that we can use and modify as a part of deep learning models for full waveform inversion.

### Inversion Net
Some commands for running inversionnet with the modified wavelet regularization

```jsx
uv run train.py -ds flatvel-a -n wavelet_test_3 -m InversionNet -g1v 1 -g2v 0 -wave 0.005  --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt --lr 0.0001 -b 256 -eb 40 -nb 5 -j 16 -d cuda
```