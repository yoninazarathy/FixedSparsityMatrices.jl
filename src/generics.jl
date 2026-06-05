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

# Element-type conversion preserves the pattern (and its backend).
function FixedSparsityMatrix{T}(A::FixedSparsityMatrix) where {T}
    return _with(convert(AbstractMatrix{T}, A.data), A.pattern)
end

# ---- broadcasting ----
# Broadcasting reads through the underlying dense data, so `A .+ 1`, `f.(A)`,
# etc. produce ordinary arrays rather than (possibly pattern-violating) wrappers.
Base.broadcastable(A::FixedSparsityMatrix) = A.data

# ---- transpose / adjoint (pattern-preserving) ----

Base.transpose(A::FixedSparsityMatrix) = _with(permutedims(A.data), permutedims(A.pattern))
Base.adjoint(A::FixedSparsityMatrix) = _with(permutedims(conj(A.data)), permutedims(A.pattern))

# ---- scaling and sign (pattern-preserving) ----

Base.:*(A::FixedSparsityMatrix, c::Number) = _with(A.data * c, A.pattern)
Base.:*(c::Number, A::FixedSparsityMatrix) = _with(c * A.data, A.pattern)
Base.:/(A::FixedSparsityMatrix, c::Number) = _with(A.data / c, A.pattern)
Base.:-(A::FixedSparsityMatrix) = _with(-A.data, A.pattern)

# ---- addition / subtraction ----
# Between two fixed-sparsity matrices the result pattern is the union of patterns.
function Base.:+(A::FixedSparsityMatrix, B::FixedSparsityMatrix)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match: $(size(A)) vs $(size(B))"))
    return _with(A.data + B.data, A.pattern .| B.pattern)
end
function Base.:-(A::FixedSparsityMatrix, B::FixedSparsityMatrix)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match: $(size(A)) vs $(size(B))"))
    return _with(A.data - B.data, A.pattern .| B.pattern)
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
