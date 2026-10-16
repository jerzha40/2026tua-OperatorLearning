# Operator Learning Research Program

## Aim

Build a long-lived research and engineering program rather than a single paper. Each method should have an explicit mathematical target, independently reproducible implementation, transparent comparison, and versioned release.

| Track | Object | First measurable milestone | Status |
| --- | --- | --- | --- |
| ResOpNet | Local signed quadrature-error operator | MLP corrector versus Simpson, Gaussian and adaptive quadrature, at matched sample counts | Specification only |
| SCHO | Spline coefficient-to-coefficient PDE solution operator | 1D integral and Poisson demos with analytic/independent references | Specification only |

Ideas for later versions: Function Family Registry, Operator Zoo, standardized tests, accuracy-and-speed arena. None of these services has been implemented.

## Work cycle

1. **Specify** function spaces, domains, sampling/coordinates, assumptions, operator and loss.
2. **Implement** minimal runnable experiments, deterministic seeds, and unit tests.
3. **Verify** against independent analytic or high-accuracy numerical solutions.
4. **Benchmark** final accuracy, failure cases, total function evaluations, wall time, memory, and offline training cost.
5. **Release** tagged versions with limitations, provenance, citations, and author credit.

Allowed project status descriptions: Idea; Mathematical specification; Prototype; Validated experiment; Release.

## Public/private boundary

The public repository is a curated distribution, not an export of the private metaEssay/DHNTeam history. Do not push private chat annotations, credentials, personal details, unapproved attachments, or private Git histories. Public explanations should be self-contained.

## Questions we must be willing to answer

- Does ResOpNet outperform an equally well-sampled *higher-order classical rule*?
- Does SCHO beat a direct spline/Galerkin solve after training cost and domain shifts?
- Do extra internal sample evaluations eliminate the claimed speed benefit?
- When does the chosen function family fail to contain unseen inputs?
- What is guaranteed by fixed independent spline coordinates, and what remains unidentifiable?
- Are low training PDE residuals reflected in independent solution error?

An external method may be better; a useful program should still be able to reproduce it, cite its authors, and compare it fairly.
