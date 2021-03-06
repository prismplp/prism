# T-PRISM-PCFG Program

This program describes how to simulate PRISM program and learn the parameters
with Probabilistic Context-free grammar (PCFG) example.
Please refer to "prism/exs/base/pdcg_c.psm", for more information related to 
the PRISM program for the PCFG.
This program realizes the parameter learning of the PCFG model  
in T-PRISM framework with the negative log likelihood loss function by uing simulated_msw/2.
The modelling part is only replaced msw/2 with simulated_msw/2.


## 1. First, make a temprary directory to store internal data files:


```
cd exs/tensor/pcfg
mkdir -p pcfg_tmp
```

## 2. Then, let us try to construct an explanation graph for PCFG and 

```
upprism pcfg.psm
```

The following grammar and goals are directly written in pcfg.psm

### Grammar:
```
Rule1: S -> X
Rule2: X -> a X a | b
```
This grammar accepts the strings such as [a,b,a], [a,a,b,a,a],  [a,a,a,b,a,a,a], ... .

### Goals:
```
pcfg([a,a,b,a,a])
pcfg([a,b,a])
```

## 3. Finaly, Maximum likelihood parameter training is carried out by the parameters using tprism.py command.
This script uses an explanation graph stored in ./pcfg_tmp/.
In the T-PRISM framework, parameter training is carried out by the stochastic gradient method (SGD) with a loss function.
Now, loss function is set as negative log likelihood (--sgd_loss nll) and parameters of SGD are configured with options: the number of epochs is ten (--max_iterate 10)
the number of samples in a minibatch is ten (--sgd_minibatch_size 10), and the learning rate is 0.01 (--sgd_learning_rate 0.01).

```
tprism.py train --internal_data_prefix ./pcfg_tmp/  --sgd_loss nll --max_iterate 10 --sgd_minibatch_size 10  --sgd_learning_rate 0.01
```

## More, detail description related to input/output files are described in run.sh

