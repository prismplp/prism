# Markov chain program

This sample program shows how to represent sum-product expressions that were redundant in the conventional method efficiently.
A PRISM program "markov_chain_msw.psm" describes Markovian chains with finite states and calculates the probability of transition from a state S to a state T with N steps.
The T-PRISM program "markov_chain.psm" is a T-PRISM for the same computation of "markov_chain_msw.psm".
Using T-PRISM's matrix representation, T-PRISM program can reduce the redundant repetition.


## 1. First, make a temprary directory to store internal data files:

```
cd exs/tensor/markov_chain
mkdir -p markov_chain_tmp
```

## 2. Then, let us try to construct an explanation graph for the Markov chain

```
upprism markov_chain.psm
```


## 3. Finaly, parameter training by maximum likelihood estimation is carried out by the parameters using tprism.py command.
This script uses an explanation graph stored in ./markov_chain_tmp/.
In the T-PRISM framework, parameter training is carried out by the stochastic gradient method (SGD) with a loss function.
Now, the loss function is set as negative log likelihood (--sgd_loss nll) and parameters of SGD are configured with options:
the maximum number of epochs is ten (--max_iterate 10),
the number of samples in a minibatch is ten (--sgd_minibatch_size 10),
and the learning rate is 0.01 (--sgd_learning_rate 0.01).

```
tprism.py train --intermediate_data_prefix ./markov_chain_tmp/  --sgd_loss nll --max_iterate 10 --sgd_minibatch_size 10  --sgd_learning_rate 0.01
```

