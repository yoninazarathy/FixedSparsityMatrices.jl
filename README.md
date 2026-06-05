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

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/yoninazarathy/FixedSparsityMatrices.jl")
```

(Registration in the General registry is planned; once registered,
`Pkg.add("FixedSparsityMatrices")` will work.)

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
same design choice `SparseArrays` makes for its structure). By default the
pattern is stored compactly as a `BitMatrix` (1 bit per entry).

## Construction

```julia
using FixedSparsityMatrices, LinearAlgebra, SparseArrays

# 1. Explicit data + boolean mask. Entries outside the mask are forced to zero
#    on construction (here the off-pattern 9.0 becomes 0.0). Arguments are copied.
A = FixedSparsityMatrix([1.0 2.0 9.0; 0.0 3.0 4.0; 5.0 0.0 6.0],
                        Bool[1 1 0; 0 1 1; 1 0 1])

# 2. Infer the pattern from the current nonzeros of a matrix.
FixedSparsityMatrix([1.0 0.0; 0.0 2.0])               # pattern = nonzeros

# 3. Take the *structural* pattern from a LinearAlgebra shape type or a sparse
#    matrix — band/triangle/stored positions, even where a structural entry is
#    numerically zero.
FixedSparsityMatrix(Diagonal([1.0, 2.0, 3.0]))        # pattern = the diagonal
FixedSparsityMatrix(Bidiagonal([1,2,3.], [4,5.], :U)) # pattern = the upper band
FixedSparsityMatrix(Tridiagonal([1,2.],[3,4,5.],[6,7.]))
FixedSparsityMatrix(UpperTriangular(ones(3, 3)))
FixedSparsityMatrix(sparse([1,2], [1,2], [1.0, 2.0]))  # pattern = stored entries
```

Supported shape types: `Diagonal`, `Bidiagonal`, `Tridiagonal`, `SymTridiagonal`,
`UpperTriangular`, `LowerTriangular`, and `SparseMatrixCSC`.

### A random instance with a given pattern

There is no special `rand` method: because construction zeros the forbidden
entries, you can build a random matrix that honors a pattern directly. (`rand(A)`
keeps its usual meaning — a random *element* of `A`.)

```julia
pat = Bool[1 1 0; 0 1 1; 1 0 1]
FixedSparsityMatrix(rand(3, 3), pat)            # U[0,1) on the pattern, zeros elsewhere
FixedSparsityMatrix(rand(Float32, 3, 3), pat)   # any element type / RNG
```

### Choosing the pattern's storage backend

By default the pattern is a `BitMatrix`. To keep a different boolean backend —
e.g. a `Matrix{Bool}`, which uses a byte per entry but has slightly faster
element access — call the parametric constructor directly:

```julia
FixedSparsityMatrix{Float64, Matrix{Float64}, Matrix{Bool}}(data, pattern)
```

## Accessors

```julia
pattern(A)   # the boolean mask of allowed positions (live internal — read-only)
parent(A)    # the underlying data matrix              (live internal — read-only)
size(A); axes(A); eltype(A)
A[i, j]      # indexing; forbidden positions read as zero
Matrix(A); collect(A); sparse(A); diag(A)
```

## Operations

Operations whose result is still constrained by the pattern **preserve** it and
return a `FixedSparsityMatrix` (keeping the pattern's storage backend):

| Operation | Result |
| --- | --- |
| `c * A`, `A * c`, `A / c` | `FixedSparsityMatrix` (same pattern) |
| `-A`, `transpose(A)`, `adjoint(A)` | `FixedSparsityMatrix` |
| `A + B`, `A - B` (both `FixedSparsityMatrix`) | `FixedSparsityMatrix` (pattern = union) |
| `copy(A)`, `zero(A)`, `FixedSparsityMatrix{T}(A)` | `FixedSparsityMatrix` |

Operations whose result is **not** constrained by the pattern degrade to ordinary
dense arrays:

| Operation | Result |
| --- | --- |
| `A * B`, `A \ b`, `A / B` (products / solves) | dense `Array` |
| `A .+ 3`, `f.(A)`, general broadcasting | dense `Array` |
| `A + M`, `A - M` with a plain matrix `M` | dense `Array` |
| `similar(A, …)` | dense `Matrix` (so generic code can fill it) |

## Semantics & gotchas

- **Construction** forces forbidden entries to `zero(T)` and defensively copies
  both arguments.
- **`setindex!`** at a forbidden position is a no-op if the value is zero and
  throws an `ArgumentError` otherwise — so `A[i,j] = 0` is always fine, but
  `A[i,j] = 5` errors when `(i,j)` is outside the pattern. This also applies to
  in-place broadcast assignment: `A .= M` enforces the pattern (and errors if `M`
  is nonzero at a forbidden position), while `A .= 0` is fine.
- **Broadcasting reads through the dense data**, so `A .+ 3`, `f.(A)`, etc. do
  **not** respect the pattern: they return a plain `Array` in which the fixed-zero
  entries are transformed too (e.g. `A .+ 3` puts `3` there). Use scalar `*`/`/`
  for pattern-preserving scaling.
- **`pattern(A)` / `parent(A)`** return the live internal arrays; treat them as
  read-only — mutating them would break the fixed-pattern invariant.

## Element types and storage requirements

`T` is unconstrained beyond needing `zero(T)` and `iszero` (it works for any
numeric element type, not just `Float64`). The data backend `M` must be a
*mutable* `AbstractMatrix` (e.g. `Matrix`, an `MMatrix`, a GPU array), since
construction and `setindex!` write into it. The matrix need not be square.

## Status

Early days (v0.1). The core type, the `AbstractMatrix` interface,
structured-type interop, and a thorough test suite are in place. Issues and PRs
welcome.
