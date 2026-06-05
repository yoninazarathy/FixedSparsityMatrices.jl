# Generic methods: conversions, broadcasting, and arithmetic.
#
# Guiding rule (the same one PDMats follows): operations that *preserve* the
# sparsity pattern return a `FixedSparsityMatrix`; everything whose result is not
# constrained by the pattern (matrix products, solves, general broadcasting)
# degrades gracefully to a plain dense array. The pattern is a property of the
# operand, not of arbitrary results computed from it.

# ---- abstract-type constructors / conversions ----

AbstractFixedSparsityMatrix(A::AbstractFixedSparsityMatrix) = A
AbstractFixedSparsityMatrix(A::AbstractMatrix) = FixedSparsityMatrix(A)

Base.Matrix(A::FixedSparsityMatrix) = Matrix(A.data)
Base.Matrix{T}(A::FixedSparsityMatrix) where {T} = Matrix{T}(A.data)
SparseArrays.sparse(A::FixedSparsityMatrix) = sparse(A.data)
LinearAlgebra.diag(A::FixedSparsityMatrix, k::Integer = 0) = diag(A.data, k)

# Element-type conversion preserves the pattern.
function FixedSparsityMatrix{T}(A::FixedSparsityMatrix) where {T}
    return FixedSparsityMatrix(convert(AbstractMatrix{T}, A.data), A.support)
end

# ---- broadcasting ----
# Broadcasting reads through the underlying dense data, so `A .+ 1`, `f.(A)`,
# etc. produce ordinary arrays rather than (possibly pattern-violating) wrappers.
Base.broadcastable(A::FixedSparsityMatrix) = A.data

# ---- transpose / adjoint (pattern-preserving) ----

Base.transpose(A::FixedSparsityMatrix) = FixedSparsityMatrix(permutedims(A.data), permutedims(A.support))
Base.adjoint(A::FixedSparsityMatrix) = FixedSparsityMatrix(permutedims(conj(A.data)), permutedims(A.support))

# ---- scaling and sign (pattern-preserving) ----

Base.:*(A::FixedSparsityMatrix, c::Number) = FixedSparsityMatrix(A.data * c, A.support)
Base.:*(c::Number, A::FixedSparsityMatrix) = FixedSparsityMatrix(c * A.data, A.support)
Base.:/(A::FixedSparsityMatrix, c::Number) = FixedSparsityMatrix(A.data / c, A.support)
Base.:-(A::FixedSparsityMatrix) = FixedSparsityMatrix(-A.data, A.support)

# ---- addition / subtraction ----
# Between two fixed-sparsity matrices the result pattern is the union of supports.
function Base.:+(A::FixedSparsityMatrix, B::FixedSparsityMatrix)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match: $(size(A)) vs $(size(B))"))
    return FixedSparsityMatrix(A.data + B.data, A.support .| B.support)
end
function Base.:-(A::FixedSparsityMatrix, B::FixedSparsityMatrix)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match: $(size(A)) vs $(size(B))"))
    return FixedSparsityMatrix(A.data - B.data, A.support .| B.support)
end
# Against an unconstrained matrix the result is unconstrained → dense.
Base.:+(A::FixedSparsityMatrix, B::AbstractMatrix) = A.data + B
Base.:+(A::AbstractMatrix, B::FixedSparsityMatrix) = A + B.data
Base.:-(A::FixedSparsityMatrix, B::AbstractMatrix) = A.data - B
Base.:-(A::AbstractMatrix, B::FixedSparsityMatrix) = A - B.data

# ---- products / solves ----
# These are intentionally NOT given bespoke methods: the generic `AbstractMatrix`
# fallbacks already degrade to dense correctly (our `similar` returns a `Matrix`,
# so `A * B`, `A \ b`, `A / B`, etc. allocate and compute dense results). Adding
# `AbstractMatrix`-typed fast paths here would clash with LinearAlgebra's
# row-vector (`Transpose`/`Adjoint`) signatures. For the small dense matrices
# this type targets, the fallbacks are sufficient.
