# Operator Learning Research Program

An open, long-term research program on **operator learning, numerical methods, and scientific machine learning**.

This is a curated public research workspace associated with the 2026 autumn term. Private exploratory notes and public deliverables are kept separate. A mathematical specification, a reproducible demo, and a validated result are different stages.

## Research tracks

**ResOpNet — Residual Operator Network.** Learn the *signed local quadrature error* while keeping the classical baseline:

\[
\widehat I_i(f)=h_i[Q(g_i)+R_\eta(S(g_i))],\quad g_i(t)=f(x_i+h_i t).
\]

A correction can use separate interior samples, but all function evaluations must be counted. See [ResOpNet specification](book/chapters/chapter4/body.tex). **Status: mathematical specification; no validated speed/accuracy advantage.**

**SCHO — Spline-Coefficient HyperOperator.** Use fixed, linearly independent input/output B-spline bases and a hypernetwork mapping their coefficient vectors. A complementary correction space can be used for SCHO-Res. PDE weak-form residuals permit training without solution labels. See [SCHO specification](book/chapters/chapter5/body.tex). **Status: proposed architecture; experiment pending.**

## Repository map

- [Program policy and proposed benchmarks](docs/program.md)
- [Book entry point](book/main.tex) and [shared LaTeX preamble](book/preamble.tex)
- [ResOpNet chapter](book/chapters/chapter4/)
- [SCHO chapter](book/chapters/chapter5/)
- [Historical note retained from initial setup](book/chapters/chapter1/)

Compile the joint notes with a TeX distribution from the book directory using: pdflatex -interaction=nonstopmode -halt-on-error main.tex. Standalone chapters can be compiled from their own directories with the same command.

## Research principles

- Compare to strong *classical numerical methods* at matched accuracy and full cost.
- Separate representation, learning, quadrature, and PDE errors.
- Distinguish unique spline coordinates from potentially non-unique trainable network weights.
- Publish source, tests, independent references, and negative results as they become available.
- Attribute related work and independent contributions; do not claim priority or empirical gains before evidence.

Listed by [uni_escuela](https://github.com/jerzha40/uni_escuela) as a public submodule. This repository is **not** a mirror of the private DHNTeam Git history.

**License:** Not yet selected. Public visibility is not an open-source license; establish licensing terms before soliciting substantial external contributions.
