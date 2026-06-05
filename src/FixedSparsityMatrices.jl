module FixedSparsityMatrices

using LinearAlgebra
using SparseArrays

export
    # Types
    AbstractFixedSparsityArray,
    AbstractFixedSparsityMatrix,
    AbstractFixedSparsityVector,
    FixedSparsityMatrix,
    FixedSparsityVector,

    # Functions
    pattern

"""
    AbstractFixedSparsityArray{T, N} <: AbstractArray{T, N}

Base type for arrays with a *fixed sparsity pattern* — a set of positions that
are allowed to be nonzero, fixed for the lifetime of the array. Entries outside
the pattern are held at exactly `zero(T)` and any attempt to set them to a
nonzero value is an error.

This is distinct from `SparseArrays` (which is about *storage* of mostly-zero
arrays and permits structural fill) and from the shape families in
`LinearAlgebra` (`Diagonal`, `Bidiagonal`, …, which encode a fixed *shape* in
the type). Here the pattern is arbitrary and carried as data.

Concrete subtypes: [`FixedSparsityMatrix`](@ref), [`FixedSparsityVector`](@ref).
"""
abstract type AbstractFixedSparsityArray{T, N} <: AbstractArray{T, N} end

"""
    AbstractFixedSparsityMatrix{T} <: AbstractFixedSparsityArray{T, 2}

The matrix (2-dimensional) case of [`AbstractFixedSparsityArray`](@ref). Because
`AbstractArray{T,2}` is `AbstractMatrix{T}`, subtypes are full `AbstractMatrix`es.

Concrete subtype: [`FixedSparsityMatrix`](@ref).
"""
abstract type AbstractFixedSparsityMatrix{T} <: AbstractFixedSparsityArray{T, 2} end

"""
    AbstractFixedSparsityVector{T} <: AbstractFixedSparsityArray{T, 1}

The vector (1-dimensional) case of [`AbstractFixedSparsityArray`](@ref). Because
`AbstractArray{T,1}` is `AbstractVector{T}`, subtypes are full `AbstractVector`s.

Concrete subtype: [`FixedSparsityVector`](@ref).
"""
abstract type AbstractFixedSparsityVector{T} <: AbstractFixedSparsityArray{T, 1} end

include("fixedsparsitymatrix.jl")
include("fixedsparsityvector.jl")
include("generics.jl")

end # module
