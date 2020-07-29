# BinaryCommitteeMachineFBP.jl

| **Documentation**                 | **Build Status**                                               |
|:---------------------------------:|:--------------------------------------------------------------:|
| [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url]  [![][codecov-img]][codecov-url] |

This package implements the Focusing Belief Propagation algorithm for
committee machines with binary weights described in the paper
*Unreasonable Effectiveness of Learning Neural Networks: From Accessible States
and Robust Ensembles to Basic Algorithmic Schemes*
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso,
Carlo Lucibello, Luca Saglietti and Riccardo Zecchina,
Proc. Natl. Acad. Sci. U.S.A. 113: E7655-E7662 (2016), [doi:10.1073/pnas.1608103113](http://dx.doi.org/10.1073/pnas.1608103113).

The code is written in [Julia](http://julialang.org). It was last tested with Julia version 1.4.

### Installation

To install the module, switch to pkg mode with the `]` key and use this command:

```
pkg> add https://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl
```

Dependencies will be installed automatically.

## Documentation

- [In-development version of the documentation][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://carlobaldassi.github.io/BinaryCommitteeMachineFBP.jl/dev

[travis-img]: https://travis-ci.org/carlobaldassi/BinaryCommitteeMachineFBP.jl.svg?branch=master
[travis-url]: https://travis-ci.org/carlobaldassi/BinaryCommitteeMachineFBP.jl

[codecov-img]: https://codecov.io/gh/carlobaldassi/BinaryCommitteeMachineFBP.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/carlobaldassi/BinaryCommitteeMachineFBP.jl
