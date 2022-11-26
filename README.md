# Full-Shape Likelihoods

This repository contains the [Montepython](https://github.com/brinckmann/montepython_public) likelihoods needed to perform full-shape analysis of current and future survey data. The likelihood can be used to analyze any of the following datasets:
- Galaxy power spectrum multipoles (P0 / P2 / P4, see [Ivanov+19](https://arxiv.org/abs/1909.05277))
- Galaxy real-space power spectrum proxy (Q0, see [Ivanov+21](https://arxiv.org/abs/2110.00006))
- Galaxy BAO rescaling parameters (AP, see [Philcox+20](https://arxiv.org/abs/2002.04035))
- Galaxy bispectrum monopole, including PNG and one-loop galaxy bias (B0, see [Philcox & Ivanov 21](https://arxiv.org/abs/2112.04515), [Cabass+22](https://arxiv.org/abs/2201.07238), [Philcox+22](https://arxiv.org/abs/2206.02800))

The likelihoods make extensive use of the [CLASS-PT](https://github.com/michalychforever/CLASS-PT) code ([Chudaykin+20](https://arxiv.org/abs/2004.10607)), which is an extension of the Boltzmann code CLASS. The one-loop bispectrum extension (which should be run at fixed cosmology, except for sigma8) additionally requires the bispectrum templates computed with the [OneLoopBispectrum](https://github.com/oliverphilcox/OneLoopBispectrum) Mathematica code.

In the ```data/``` directory, we include measurements of each of the above data-sets for the BOSS survey. Note that power spectra and bispectra do not include the survey window function; these are computed with the estimators of [Philcox 20](https://arxiv.org/abs/2012.09389), and [Philcox 21](https://arxiv.org/abs/2107.06287), as in the [Spectra-Without-Windows](https://github.com/oliverphilcox/Spectra-Without-Windows) repository. We provide a sample ```.param``` file in the ```input/``` directory; this will reproduce the full BOSS analysis of [Philcox & Ivanov 21](https://arxiv.org/abs/2112.04515).

## Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Columbia / Simons Foundation)
- Mikhail Ivanov (IAS)
- Giovanni Cabass (IAS)

***Newly added features:*** *one-loop bispectrum templates and redshift-dependent scale cuts*
