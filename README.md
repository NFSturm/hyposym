# HypoSym
Hypothesis-guided n-ary relation extraction with symbols.

This repository implements a symbolic reasoning engine to be applied on the Drug Effects Interaction task proposed by [Tiktinsky et al. (2022)](https://arxiv.org/abs/2205.02289).

The approach is based on matching keyword expressions in knowledge bases and reasoning operates in two steps:

- (I) Match derived drug combinations against those in the drug combination knowledge base
- (II) Match derived keyword expressions against those in the keyword expressions knowledge base
