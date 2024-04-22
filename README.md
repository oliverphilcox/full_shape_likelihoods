# Full-Shape Likelihoods

This repository contains the [Montepython](https://github.com/brinckmann/montepython_public) likelihoods needed to perform full-shape analysis of current and future survey data. The likelihood can be used to analyze any of the following datasets:
- Galaxy power spectrum multipoles (P0 / P2 / P4, see [Ivanov+19](https://arxiv.org/abs/1909.05277))
- Galaxy real-space power spectrum proxy (Q0, see [Ivanov+21](https://arxiv.org/abs/2110.00006))
- Galaxy BAO rescaling parameters (AP, see [Philcox+20](https://arxiv.org/abs/2002.04035))
- Galaxy bispectrum multipoles (B0 / B2 / B4, see [Philcox & Ivanov 21](https://arxiv.org/abs/2112.04515), [Ivanov+22](https://arxiv.org/abs/2302.04414)) including PNG (see [Cabass+22](https://arxiv.org/abs/2201.07238), [Cabass+24](https://arxiv.org/abs/2404.01894)).

The likelihoods make extensive use of the [CLASS-PT](https://github.com/michalychforever/CLASS-PT) code ([Chudaykin+20](https://arxiv.org/abs/2004.10607)), which is an extension of the Boltzmann code CLASS. Although the one-loop bispectrum templates have been computed (see [OneLoopBispectrum](https://github.com/oliverphilcox/OneLoopBispectrum), [Philcox+22](https://arxiv.org/abs/2206.02800)), these are not currently included in the likelihoods since they add little constraining power.

We can analyze both (conventional) Gaussian initial conditions, as well as modifications for primordial non-Gaussianity. In particular, we include a full implementation of the equilateral and orthogonal templates, which can be turned on in the ```.param``` file. Optionally, cosmological collider non-Gaussianity can be included (following [Cabass+24](https://arxiv.org/abs/2404.01894)). This requires a set of interpolation tables for the one-loop power spectrum (P_12), which can be provided on request.

In the ```data/``` directory, we include measurements of each of the above data-sets for the BOSS survey. Note that power spectra and bispectra do not include the survey window function; these are computed with the estimators of [Philcox 20](https://arxiv.org/abs/2012.09389), and [Philcox 21](https://arxiv.org/abs/2107.06287), as in the [PolyBin3D](https://github.com/oliverphilcox/PolyBin3D) repository. We provide a sample ```.param``` file in the ```input/``` directory; this will reproduce a full BOSS analysis similar to [Philcox & Ivanov 21](https://arxiv.org/abs/2112.04515).

## Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Columbia / Simons Foundation)
- Mikhail Ivanov (MIT)
- Giovanni Cabass (Ruder Boskovic Institute, Zagreb)

***Newly added features:*** cosmological collider non-Gaussianity and general streamlining*
