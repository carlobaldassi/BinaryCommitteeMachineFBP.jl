# BinaryCommitteeMachineFBP.jl

| **Documentation**                       | **Build Status**                                                                                |
|:---------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-latest-img]][docs-latest-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] |

This package implements the Focusing Belief Propagation algorithm for
committee machines with binary weights described in the paper
[Unreasonable Effectiveness of Learning Neural Networks: From Accessible States and Robust Ensembles to Basic Algorithmic Schemes](http://arxiv.org/abs/1605.06444)
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso, Carlo Lucibello, Luca Saglietti and Riccardo Zecchina.

The code is written in [Julia](http://julialang.org).

The package is tested against Julia `0.4` and *current* `0.5-dev` on Linux, OS X, and Windows.

### Installation

To install the module, use this command from within Julia:

```
julia> Pkg.clone("https://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl")
```

Dependencies will be installed automatically.

## Documentation

- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://carlobaldassi.github.io/BinaryCommitteeMachineFBP.jl/latest

[travis-img]: https://travis-ci.org/carlobaldassi/BinaryCommitteeMachineFBP.jl.svg?branch=master
[travis-url]: https://travis-ci.org/carlobaldassi/BinaryCommitteeMachineFBP.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/aeclj3cs8c2l0tvt/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/carlobaldassi/binarycommitteemachinefbp-jl/branch/master

[codecov-img]: https://codecov.io/gh/carlobaldassi/BinaryCommitteeMachineFBP.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/carlobaldassi/BinaryCommitteeMachineFBP.jl
