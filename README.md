# FixedSparsityMatrices.jl

[![CI](https://github.com/yoninazarathy/FixedSparsityMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yoninazarathy/FixedSparsityMatrices.jl/actions/workflows/CI.yml)

A small Julia package providing `FixedSparsityMatrix` — a dense matrix paired
with a **fixed sparsity pattern**: a set of positions allowed to be nonzero,
fixed for the lifetime of the matrix. Entries outside the pattern are held at
exactly zero and *cannot* be set to a nonzero value.

It subtypes `AbstractMatrix`, so it works with the rest of the linear-algebra
ecosystem, and it interoperates with the structured matrix types from
`LinearAlgebra` (`Diagonal`, `Bidiagonal`, `Tridiagonal`, `UpperTriangular`, …)
and with `SparseArrays`.

## Why not `SparseArrays` or `Diagonal`/`Bidiagonal`/…?

- **`SparseArrays`** is about *storage* of mostly-zero matrices, and it *permits*
  structural fill — writing a new nonzero into a previously-zero position is
  allowed (it changes the structure). `FixedSparsityMatrix` does the opposite:
  the pattern is fixed and enforced, so forbidden positions can never become
  nonzero. It also stores densely (intended for small/medium matrices), so the
  pattern — not storage efficiency — is the point.
- **`Diagonal`, `Bidiagonal`, `UpperTriangular`, …** each represent one
  *specific* zero structure, and there is a separate Julia type for each one — so
  they only cover those standard shapes. `FixedSparsityMatrix` instead holds the
  pattern as an ordinary value (the `pattern` mask), so a single type can
  represent *any* arrangement of fixed zeros, including irregular ones that have
  no dedicated type. When your structure does happen to be one of the standard
  shapes, you can construct a `FixedSparsityMatrix` straight from it — e.g.
  `FixedSparsityMatrix(Diagonal([1.0, 2.0, 3.0]))`.

## Type

```julia
abstract type AbstractFixedSparsityMatrix{T} <: AbstractMatrix{T} end

struct FixedSparsityMatrix{T, M<:AbstractMatrix{T}, P<:AbstractMatrix{Bool}} <: AbstractFixedSparsityMatrix{T}
    data::M       # values (forbidden entries held at zero)
    pattern::P    # boolean mask: true where entries may be nonzero
end
```

The pattern lives in the `pattern` *field*, not the type — so distinct patterns
do not produce distinct types and there is no recompilation per pattern (the
same design choice `SparseArrays` makes for its structure).

## Usage

```julia
using FixedSparsityMatrices, LinearAlgebra

# Explicit pattern (the off-pattern 9.0 is forced to zero on construction):
A = FixedSparsityMatrix([1.0 2.0 9.0; 0.0 3.0 4.0; 5.0 0.0 6.0],
                        Bool[1 1 0; 0 1 1; 1 0 1])
A[1, 3]            # 0.0
A[1, 2] = 5.0      # allowed
A[1, 3] = 7.0      # ERROR: entry (1,3) is fixed to zero

pattern(A)         # the boolean pattern (allowed positions)
Matrix(A)          # dense copy
2A                 # scaling preserves the pattern (returns a FixedSparsityMatrix)
A * A              # products degrade to a plain dense Matrix
A .+ 3             # broadcasting ignores the pattern → plain dense Array;
                   #   the fixed-zero entries become 3 too (0 + 3)

# Infer the pattern from current nonzeros, or take it from a structured type:
FixedSparsityMatrix([1.0 0.0; 0.0 2.0])              # pattern = nonzeros
FixedSparsityMatrix(Bidiagonal([1,2,3.], [4,5.], :U)) # pattern = the band
FixedSparsityMatrix(Diagonal([1.0, 2.0, 3.0]))        # pattern = the diagonal
```

## Semantics at a glance

- **Construction** forces forbidden entries to `zero(T)` (defensively copying its
  arguments).
- **`setindex!`** on a forbidden position is a no-op if the value is zero and an
  `ArgumentError` otherwise.
- **Pattern-preserving** operations (`*`/`/` by a scalar, unary `-`, `transpose`,
  `adjoint`, and `+`/`-` between two `FixedSparsityMatrix`es — pattern is the
  union) return a `FixedSparsityMatrix`.
- **Unconstrained** results (matrix products, solves, general broadcasting,
  `similar`) degrade to ordinary dense arrays. In particular, broadcasting reads
  through the underlying dense data, so `A .+ 3`, `f.(A)`, etc. do **not** respect
  the pattern: they return a plain `Array` in which the fixed-zero entries are
  transformed too (e.g. `A .+ 3` puts `3` at those positions). Use scalar `*`/`/`
  for pattern-preserving scaling.

## Status

Early days (v0.1). The core type, interface, structured-type interop, and tests
are in place. Issues and PRs welcome.
