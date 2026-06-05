module FixedSparsityMatrices

using LinearAlgebra
using SparseArrays

export
    # Types
    AbstractFixedSparsityMatrix,
    FixedSparsityMatrix,

    # Functions
    pattern

"""
    AbstractFixedSparsityMatrix{T} <: AbstractMatrix{T}

Base type for matrices with a *fixed sparsity pattern* — a set of positions that
are allowed to be nonzero, fixed for the lifetime of the matrix.
Entries outside the pattern are held at exactly `zero(T)` and any attempt to set
them to a nonzero value is an error.

This is distinct from `SparseArrays` (which is about *storage* of mostly-zero
matrices and permits structural fill) and from the shape families in
`LinearAlgebra` (`Diagonal`, `Bidiagonal`, …, which encode a fixed *shape* in
the type). Here the pattern is arbitrary and carried as data.

Concrete subtype: [`FixedSparsityMatrix`](@ref).
"""
abstract type AbstractFixedSparsityMatrix{T} <: AbstractMatrix{T} end

include("fixedsparsitymatrix.jl")
include("generics.jl")

end # module
