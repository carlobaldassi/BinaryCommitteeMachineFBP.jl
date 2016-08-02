# BinaryCommitteeMachineFBP.jl documentation

This package implements the Focusing Belief Propagation algorithm for
committee machines with binary weights described in the paper
[Unreasonable Effectiveness of Learning Neural Nets: Accessible States and Robust Ensembles](http://arxiv.org/abs/1605.06444)
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso, Carlo Lucibello, Luca Saglietti and Riccardo Zecchina.

The package is tested against Julia `0.4` and *current* `0.5-dev` on Linux, OS X, and Windows.

### Installation

To install the module, use this command from within Julia:

```
julia> Pkg.clone("https://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl")
```

Dependencies will be installed automatically.

### Usage

The module is loaded as any other Julia module:

```
julia> using BinaryCommitteeMachineFBP
```

The code provides a main function, [`focusingBP`](@ref), and some auxiliary functions and types, documented below.

```@docs
focusingBP
```

#### Focusing protocols

```@docs
FocusingProtocol
```

```@docs
StandardReinforcement
```

```@docs
Scoping
```

```@docs
PseudoReinforcement
```

```@docs
FreeScoping
```

### Reading and writing messages files

```@docs
read_messages
```

```@docs
write_messages
```

