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

# Template instantiation: an all-zeros, `T`-valued matrix over a given sparsity
# pattern (then fill the allowed entries with `setindex!`). The `{T}` is what
# marks the boolean matrix as a *pattern* rather than data to infer nonzeros from.
FixedSparsityMatrix{T}(pattern::AbstractMatrix{Bool}) where {T} =
    FixedSparsityMatrix(zeros(T, size(pattern)), pattern)

# Disambiguate the exotic Bool-valued FixedSparsityMatrix (which is both a
# FixedSparsityMatrix and an AbstractMatrix{Bool}) in favour of element conversion.
FixedSparsityMatrix{T}(A::FixedSparsityMatrix{Bool}) where {T} =
    _with(convert(AbstractMatrix{T}, A.data), A.pattern)

# ---- broadcasting ----
# Broadcasting reads through the underlying dense data, so `A .+ 1`, `f.(A)`,
# etc. produce ordinary arrays rather than (possibly pattern-violating) wrappers.
Base.broadcastable(A::FixedSparsityMatrix) = A.data

# ---- transpose / adjoint (pattern-preserving) ----

# transpose/adjoint produce a fresh (permuted) pattern; scaling/sign keep and
# share the same pattern object. All produce freshly-owned data, so use `_wrap`.
Base.transpose(A::FixedSparsityMatrix) = _wrap(permutedims(A.data), permutedims(A.pattern))
Base.adjoint(A::FixedSparsityMatrix) = _wrap(permutedims(conj(A.data)), permutedims(A.pattern))

# ---- scaling and sign (pattern-preserving; pattern shared) ----

Base.:*(A::FixedSparsityMatrix, c::Number) = _wrap(A.data * c, A.pattern)
Base.:*(c::Number, A::FixedSparsityMatrix) = _wrap(c * A.data, A.pattern)
Base.:/(A::FixedSparsityMatrix, c::Number) = _wrap(A.data / c, A.pattern)
Base.:-(A::FixedSparsityMatrix) = _wrap(-A.data, A.pattern)

# ---- addition / subtraction ----
# Between two fixed-sparsity matrices the result pattern is the union of patterns.
function Base.:+(A::FixedSparsityMatrix, B::FixedSparsityMatrix)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match: $(size(A)) vs $(size(B))"))
    return _wrap(A.data + B.data, A.pattern .| B.pattern)
end
function Base.:-(A::FixedSparsityMatrix, B::FixedSparsityMatrix)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match: $(size(A)) vs $(size(B))"))
    return _wrap(A.data - B.data, A.pattern .| B.pattern)
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
