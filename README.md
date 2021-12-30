# StochasticFokkerPlanck
Spiking neural network can be studied using mean-field theory in the contest of population density approach.
The time evolution of the distribution of membrane potential p(v,t) follows a Fokker-Planck equation. 
In a recent paper we extend this by including self-consisest finite size noise that is capable to capture even non-linear and critical regimes.
In this repository you can find the python script for the integration of the Fokker-Planck equation that are essentially the one relaesed in:
https://github.com/neuromethods/fokker-planck-based-spike-rate-models
With the necessary modification to include finite size effect.
Markovian embending of the noise is also implemented.

For reference to the theory:

### Requuired modules
The scripts run on  Python 3.x and require the following moudles
* scipy 
* numpy
* mpmath
* numba (for fast computation)
