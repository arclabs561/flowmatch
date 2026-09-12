# flowmatch scope

`flowmatch` is a small set of flow-matching primitives and demonstrations, not
a reproduction of a flow-matching paper or a general generative-modeling
framework.

## Current state

- **Conditional FM** uses the linear path $x_t = (1-t)x_0 + t y_j$ with target
  $u_t = y_j - x_0$. The baseline field is conditioned on the selected target
  $y_j$ during training and sampling.
- **RFM** provides minibatch pairing helpers and a straight-line training
  baseline.
- **ODE sampling** provides fixed-step Euler and Heun integration for the
  Euclidean baseline.
- **Experimental Riemannian support** is feature-gated through `riemannian` and
  provides training and ODE-integration primitives over `skel::Manifold`.
- **Experimental discrete support** provides CTMC probability-path, schedule,
  rate-matrix, and generator-validation primitives. It is available in the
  default module set, not a separate feature.
- **Burn support** is an optional training backend; the default public API is
  ndarray-based.

## Limits

- The included models are intentionally small linear or example-local models.
  Their fixed-seed metrics are demonstrations, not benchmark results or claims
  of general sample quality.
- RFM minibatch pairings are local coupling heuristics. Assigned-target MSE
  does not establish unconditional generation or a general coupling advantage.
- The CTMC module does not provide a learned discrete generative model or a
  full discrete-FM training workflow.
- Riemannian support is limited to the exposed manifold operations and examples;
  it is not a general geometry or equivariant modeling system.
- There is no CLI, experiment runner, or SE(3)-equivariant model.

## References

- Lipman et al., *Flow Matching for Generative Modeling* (arXiv:2210.02747)
- Lipman et al., *Flow Matching Guide and Code* (arXiv:2412.06264)
- Gat et al., *Discrete Flow Matching* (NeurIPS 2024)
- Chen & Lipman, *Riemannian Flow Matching on General Geometries*
  (arXiv:2302.03660)
