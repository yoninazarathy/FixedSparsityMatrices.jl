# The concrete dense-backed fixed-sparsity matrix type, its constructors, and the
# AbstractArray interface (including the zero-enforcing `setindex!`).

"""
    FixedSparsityMatrix(data, support)
    FixedSparsityMatrix(A)

A dense matrix `data` paired with a boolean `support` mask marking the entries
that are allowed to be nonzero. Entries where `support` is `false` are forced to
`zero(T)` at construction and can never be set to a nonzero value afterwards
(`setindex!` throws); reading them returns zero.

The type is parameterized on the storage types of both fields,
`FixedSparsityMatrix{T, M<:AbstractMatrix{T}, S<:AbstractMatrix{Bool}}`, so the
pattern is *data* (a field), not part of the type.

# Constructors
- `FixedSparsityMatrix(data, support)` — explicit mask.
- `FixedSparsityMatrix(A::AbstractMatrix)` — infer the support from the current
  nonzeros of `A`.
- `FixedSparsityMatrix(A::Diagonal | Bidiagonal | Tridiagonal | SymTridiagonal |
  UpperTriangular | LowerTriangular | SparseMatrixCSC)` — take the support from
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
struct FixedSparsityMatrix{T, M <: AbstractMatrix{T}, S <: AbstractMatrix{Bool}} <: AbstractFixedSparsityMatrix{T}
    data::M
    support::S

    function FixedSparsityMatrix{T, M, S}(data::AbstractMatrix, support::AbstractMatrix) where {T, M <: AbstractMatrix{T}, S <: AbstractMatrix{Bool}}
        axes(data) == axes(support) || throw(DimensionMismatch(
            "data has axes $(axes(data)) but support has axes $(axes(support))"))
        d = convert(M, copy(data))
        s = convert(S, copy(support))
        @inbounds for i in eachindex(d, s)
            s[i] || (d[i] = zero(T))
        end
        return new{T, M, S}(d, s)
    end
end

# Main outer constructor: capture the concrete field types.
function FixedSparsityMatrix(data::AbstractMatrix{T}, support::AbstractMatrix{Bool}) where {T}
    return FixedSparsityMatrix{T, typeof(data), typeof(support)}(data, support)
end

# Infer the support from the current nonzero entries of `A`.
FixedSparsityMatrix(A::AbstractMatrix) = FixedSparsityMatrix(A, .!iszero.(A))

# No-op / copy from another fixed-sparsity matrix.
FixedSparsityMatrix(A::FixedSparsityMatrix) = FixedSparsityMatrix(copy(A.data), copy(A.support))

# ---- Construction from LinearAlgebra shape families (structural support) ----

function _support_from(A::AbstractMatrix, isstructural)
    m, n = size(A)
    s = falses(m, n)
    @inbounds for j in 1:n, i in 1:m
        s[i, j] = isstructural(i, j)
    end
    return s
end

FixedSparsityMatrix(A::Diagonal)        = FixedSparsityMatrix(Matrix(A), _support_from(A, (i, j) -> i == j))
FixedSparsityMatrix(A::SymTridiagonal)  = FixedSparsityMatrix(Matrix(A), _support_from(A, (i, j) -> abs(i - j) <= 1))
FixedSparsityMatrix(A::Tridiagonal)     = FixedSparsityMatrix(Matrix(A), _support_from(A, (i, j) -> abs(i - j) <= 1))
FixedSparsityMatrix(A::UpperTriangular) = FixedSparsityMatrix(Matrix(A), _support_from(A, (i, j) -> i <= j))
FixedSparsityMatrix(A::LowerTriangular) = FixedSparsityMatrix(Matrix(A), _support_from(A, (i, j) -> i >= j))

function FixedSparsityMatrix(A::Bidiagonal)
    isupper = A.uplo == 'U'
    rule = isupper ? (i, j) -> i == j || j == i + 1 : (i, j) -> i == j || j == i - 1
    return FixedSparsityMatrix(Matrix(A), _support_from(A, rule))
end

# From a sparse matrix: take the *stored* structure as the support (this may
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

The support mask: `true` exactly at the positions allowed to be nonzero.
"""
pattern(A::FixedSparsityMatrix) = A.support

"""
    parent(A::FixedSparsityMatrix)

The underlying data matrix (with forbidden entries held at zero).
"""
Base.parent(A::FixedSparsityMatrix) = A.data

# ---- AbstractArray interface ----

Base.size(A::FixedSparsityMatrix) = size(A.data)
Base.axes(A::FixedSparsityMatrix) = axes(A.data)
Base.IndexStyle(::Type{<:FixedSparsityMatrix{T, M}}) where {T, M} = IndexStyle(M)

Base.@propagate_inbounds Base.getindex(A::FixedSparsityMatrix, i::Int) = A.data[i]
Base.@propagate_inbounds Base.getindex(A::FixedSparsityMatrix, i::Int, j::Int) = A.data[i, j]

Base.@propagate_inbounds function Base.setindex!(A::FixedSparsityMatrix, v, i::Int)
    if A.support[i]
        A.data[i] = v
    else
        iszero(v) || throw(ArgumentError(
            "cannot set entry (linear index $i) to a nonzero value; it is fixed to zero by the sparsity pattern"))
    end
    return v
end

Base.@propagate_inbounds function Base.setindex!(A::FixedSparsityMatrix, v, i::Int, j::Int)
    if A.support[i, j]
        A.data[i, j] = v
    else
        iszero(v) || throw(ArgumentError(
            "cannot set entry ($i, $j) to a nonzero value; it is fixed to zero by the sparsity pattern"))
    end
    return v
end

# Generic algorithms that `similar` then fill every entry would hit forbidden
# positions, so `similar` deliberately degrades to a plain dense `Matrix`.
Base.similar(::FixedSparsityMatrix, ::Type{Tv}, dims::Dims) where {Tv} = Matrix{Tv}(undef, dims)

Base.copy(A::FixedSparsityMatrix) = FixedSparsityMatrix(copy(A.data), copy(A.support))
Base.zero(A::FixedSparsityMatrix) = FixedSparsityMatrix(zero(A.data), copy(A.support))
