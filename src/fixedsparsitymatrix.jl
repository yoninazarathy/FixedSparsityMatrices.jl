# The concrete dense-backed fixed-sparsity matrix type, its constructors, and the
# AbstractArray interface (including the zero-enforcing `setindex!`).

"""
    FixedSparsityMatrix(data, pattern)
    FixedSparsityMatrix(A)

A dense matrix `data` paired with a boolean `pattern` mask marking the entries
that are allowed to be nonzero. Entries where `pattern` is `false` are forced to
`zero(T)` at construction and can never be set to a nonzero value afterwards
(`setindex!` throws); reading them returns zero.

The type is parameterized on the storage types of both fields,
`FixedSparsityMatrix{T, M<:AbstractMatrix{T}, P<:AbstractMatrix{Bool}}`, so the
pattern is *data* (a field), not part of the type.

By default the pattern is stored compactly as a `BitMatrix` (1 bit per entry).
To keep a different boolean backend instead — e.g. a `Matrix{Bool}`, which uses
a byte per entry but has slightly faster element access — call the parametric
constructor directly: `FixedSparsityMatrix{T, M, Matrix{Bool}}(data, pattern)`.

# Constructors
- `FixedSparsityMatrix(data, pattern)` — explicit mask (stored as a `BitMatrix`).
- `FixedSparsityMatrix(A::AbstractMatrix)` — infer the pattern from the current
  nonzeros of `A`.
- `FixedSparsityMatrix(A::Diagonal | Bidiagonal | Tridiagonal | SymTridiagonal |
  UpperTriangular | LowerTriangular | SparseMatrixCSC)` — take the pattern from
  the *structural* pattern of `A` (band/triangle/stored positions), independent
  of whether individual structural entries happen to be zero.

# Examples
```jldoctest
julia> A = FixedSparsityMatrix([1.0 2.0; 0.0 3.0], Bool[1 1; 0 1]);

julia> A[2, 1]
0.0

julia> A[1, 2] = 5.0; A[1, 2]
5.0

julia> A[2, 1] = 7.0
ERROR: ArgumentError: cannot set entry (2, 1) to a nonzero value; it is fixed to zero by the sparsity pattern
```
"""
struct FixedSparsityMatrix{T, M <: AbstractMatrix{T}, P <: AbstractMatrix{Bool}} <: AbstractFixedSparsityMatrix{T}
    data::M
    pattern::P

    function FixedSparsityMatrix{T, M, P}(data::AbstractMatrix, pattern::AbstractMatrix) where {T, M <: AbstractMatrix{T}, P <: AbstractMatrix{Bool}}
        axes(data) == axes(pattern) || throw(DimensionMismatch(
            "data has axes $(axes(data)) but pattern has axes $(axes(pattern))"))
        d = convert(M, copy(data))
        p = convert(P, copy(pattern))
        @inbounds for i in eachindex(d, p)
            p[i] || (d[i] = zero(T))
        end
        return new{T, M, P}(d, p)
    end

    # Trusted, allocation-free builder used internally by pattern-preserving
    # operations. The caller guarantees that `data` is freshly owned and already
    # zero outside `pattern`, and that `pattern` is an owned, read-only mask (which
    # may therefore be *shared* between matrices). No copying, re-zeroing, or checks.
    global _wrap(data::M, pattern::P) where {T, M <: AbstractMatrix{T}, P <: AbstractMatrix{Bool}} =
        new{T, M, P}(data, pattern)
end

# Main outer constructor: store the pattern compactly as a `BitMatrix` by default.
function FixedSparsityMatrix(data::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {T}
    return FixedSparsityMatrix{T, typeof(data), BitMatrix}(data, pattern)
end

# Internal: faithful (copying) rebuild, preserving whatever pattern backend
# `pattern` has. Used where an input may be borrowed/aliased (e.g. element-type
# conversion), so it goes through the checked, copying constructor. Operations
# that produce freshly-owned data with a known pattern use `_wrap` instead.
_with(data::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {T} =
    FixedSparsityMatrix{T, typeof(data), typeof(pattern)}(data, pattern)

# Infer the pattern from the current nonzero entries of `A`.
FixedSparsityMatrix(A::AbstractMatrix) = FixedSparsityMatrix(A, .!iszero.(A))

# Copy from another fixed-sparsity matrix (preserves its pattern backend).
FixedSparsityMatrix(A::FixedSparsityMatrix) = copy(A)

# ---- Construction from LinearAlgebra shape families (structural pattern) ----

function _pattern_from(A::AbstractMatrix, isstructural)
    m, n = size(A)
    s = falses(m, n)
    @inbounds for j in 1:n, i in 1:m
        s[i, j] = isstructural(i, j)
    end
    return s
end

FixedSparsityMatrix(A::Diagonal)        = FixedSparsityMatrix(Matrix(A), _pattern_from(A, (i, j) -> i == j))
FixedSparsityMatrix(A::SymTridiagonal)  = FixedSparsityMatrix(Matrix(A), _pattern_from(A, (i, j) -> abs(i - j) <= 1))
FixedSparsityMatrix(A::Tridiagonal)     = FixedSparsityMatrix(Matrix(A), _pattern_from(A, (i, j) -> abs(i - j) <= 1))
FixedSparsityMatrix(A::UpperTriangular) = FixedSparsityMatrix(Matrix(A), _pattern_from(A, (i, j) -> i <= j))
FixedSparsityMatrix(A::LowerTriangular) = FixedSparsityMatrix(Matrix(A), _pattern_from(A, (i, j) -> i >= j))

function FixedSparsityMatrix(A::Bidiagonal)
    isupper = A.uplo == 'U'
    rule = isupper ? (i, j) -> i == j || j == i + 1 : (i, j) -> i == j || j == i - 1
    return FixedSparsityMatrix(Matrix(A), _pattern_from(A, rule))
end

# From a sparse matrix: take the *stored* structure as the pattern (this may
# include explicitly-stored zeros, which is the point — it is the fixed pattern).
function FixedSparsityMatrix(A::SparseMatrixCSC)
    s = falses(size(A)...)
    rows = rowvals(A)
    @inbounds for j in axes(A, 2), idx in nzrange(A, j)
        s[rows[idx], j] = true
    end
    return FixedSparsityMatrix(Matrix(A), s)
end

# ---- Accessors ----

"""
    pattern(A::FixedSparsityMatrix) -> AbstractMatrix{Bool}

The pattern mask: `true` exactly at the positions allowed to be nonzero. Returns
the matrix's live internal mask — treat it as read-only (mutating it would break
the fixed-pattern invariant).
"""
pattern(A::FixedSparsityMatrix) = A.pattern

"""
    parent(A::FixedSparsityMatrix)

The underlying data matrix (with forbidden entries held at zero). Returns the
live internal storage — treat it as read-only.
"""
Base.parent(A::FixedSparsityMatrix) = A.data

# ---- AbstractArray interface ----

Base.size(A::FixedSparsityMatrix) = size(A.data)
Base.axes(A::FixedSparsityMatrix) = axes(A.data)
Base.IndexStyle(::Type{<:FixedSparsityMatrix{T, M}}) where {T, M} = IndexStyle(M)

Base.@propagate_inbounds Base.getindex(A::FixedSparsityMatrix, i::Int) = A.data[i]
Base.@propagate_inbounds Base.getindex(A::FixedSparsityMatrix, i::Int, j::Int) = A.data[i, j]

Base.@propagate_inbounds function Base.setindex!(A::FixedSparsityMatrix, v, i::Int)
    if A.pattern[i]
        A.data[i] = v
    else
        iszero(v) || throw(ArgumentError(
            "cannot set entry (linear index $i) to a nonzero value; it is fixed to zero by the sparsity pattern"))
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(A::FixedSparsityMatrix, v, i::Int, j::Int)
    if A.pattern[i, j]
        A.data[i, j] = v
    else
        iszero(v) || throw(ArgumentError(
            "cannot set entry ($i, $j) to a nonzero value; it is fixed to zero by the sparsity pattern"))
    end
    return v
end

# Generic algorithms that `similar` then fill every entry would hit forbidden
# positions, so `similar` deliberately degrades to a plain dense array. We honor
# the requested `dims` rank (e.g. a 1-D slice `A[:, j]` asks for a vector), so use
# `Array` rather than `Matrix` here.
Base.similar(::FixedSparsityMatrix, ::Type{Tv}, dims::Dims) where {Tv} = Array{Tv}(undef, dims)

# Independent data, shared (read-only) pattern.
Base.copy(A::FixedSparsityMatrix) = _wrap(copy(A.data), A.pattern)
Base.zero(A::FixedSparsityMatrix) = _wrap(zero(A.data), A.pattern)
