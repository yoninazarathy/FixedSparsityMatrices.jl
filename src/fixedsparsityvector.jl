# The concrete dense-backed fixed-sparsity vector type, its constructors, the
# AbstractArray interface (including the zero-enforcing `setindex!`), and the
# pattern-preserving generic methods. The vector mirrors `FixedSparsityMatrix`;
# see that file and `generics.jl` for the design rationale shared by both.

"""
    FixedSparsityVector(data, pattern)
    FixedSparsityVector(v)

A dense vector `data` paired with a boolean `pattern` mask marking the entries
that are allowed to be nonzero. Entries where `pattern` is `false` are forced to
`zero(T)` at construction and can never be set to a nonzero value afterwards
(`setindex!` throws); reading them returns zero.

The vector analogue of [`FixedSparsityMatrix`](@ref). It is parameterized on the
storage types of both fields,
`FixedSparsityVector{T, V<:AbstractVector{T}, P<:AbstractVector{Bool}}`, so the
pattern is *data* (a field), not part of the type.

By default the pattern is stored compactly as a `BitVector` (1 bit per entry).
To keep a different boolean backend instead — e.g. a `Vector{Bool}` — call the
parametric constructor directly: `FixedSparsityVector{T, V, Vector{Bool}}(data, pattern)`.

# Constructors
- `FixedSparsityVector(data, pattern)` — explicit mask (stored as a `BitVector`).
- `FixedSparsityVector(v::AbstractVector)` — infer the pattern from the current
  nonzeros of `v`.
- `FixedSparsityVector{T}(pattern::AbstractVector{Bool})` — an all-zeros, `T`-valued
  vector over the given pattern (a "template" to fill via `setindex!`).

# Examples
```jldoctest
julia> v = FixedSparsityVector([1.0, 0.0, 3.0], Bool[1, 0, 1]);

julia> v[2]
0.0

julia> v[1] = 5.0; v[1]
5.0

julia> v[2] = 7.0
ERROR: ArgumentError: cannot set entry (linear index 2) to a nonzero value; it is fixed to zero by the sparsity pattern
```
"""
struct FixedSparsityVector{T, V <: AbstractVector{T}, P <: AbstractVector{Bool}} <: AbstractFixedSparsityVector{T}
    data::V
    pattern::P

    function FixedSparsityVector{T, V, P}(data::AbstractVector, pattern::AbstractVector) where {T, V <: AbstractVector{T}, P <: AbstractVector{Bool}}
        axes(data) == axes(pattern) || throw(DimensionMismatch(
            "data has axes $(axes(data)) but pattern has axes $(axes(pattern))"))
        d = convert(V, copy(data))
        p = convert(P, copy(pattern))
        @inbounds for i in eachindex(d, p)
            p[i] || (d[i] = zero(T))
        end
        return new{T, V, P}(d, p)
    end

    # Trusted, allocation-free builder used internally by pattern-preserving
    # operations. The caller guarantees that `data` is freshly owned and already
    # zero outside `pattern`, and that `pattern` is an owned, read-only mask (which
    # may therefore be *shared* between vectors). No copying, re-zeroing, or checks.
    global _wrapvec(data::V, pattern::P) where {T, V <: AbstractVector{T}, P <: AbstractVector{Bool}} =
        new{T, V, P}(data, pattern)
end

# Main outer constructor: store the pattern compactly as a `BitVector` by default.
function FixedSparsityVector(data::AbstractVector{T}, pattern::AbstractVector{Bool}) where {T}
    return FixedSparsityVector{T, typeof(data), BitVector}(data, pattern)
end

# Internal: faithful (copying) rebuild, preserving whatever pattern backend
# `pattern` has. Used where an input may be borrowed/aliased.
_withvec(data::AbstractVector{T}, pattern::AbstractVector{Bool}) where {T} =
    FixedSparsityVector{T, typeof(data), typeof(pattern)}(data, pattern)

# Infer the pattern from the current nonzero entries of `v`.
FixedSparsityVector(v::AbstractVector) = FixedSparsityVector(v, .!iszero.(v))

# Copy from another fixed-sparsity vector (preserves its pattern backend).
FixedSparsityVector(v::FixedSparsityVector) = copy(v)

# ---- abstract-type constructors / conversions ----

AbstractFixedSparsityVector(v::AbstractFixedSparsityVector) = v
AbstractFixedSparsityVector(v::AbstractVector) = FixedSparsityVector(v)

Base.Vector(v::FixedSparsityVector) = Vector(v.data)
Base.Vector{T}(v::FixedSparsityVector) where {T} = Vector{T}(v.data)
SparseArrays.sparse(v::FixedSparsityVector) = sparse(v.data)

# Element-type conversion preserves the pattern (and its backend).
function FixedSparsityVector{T}(v::FixedSparsityVector) where {T}
    return _withvec(convert(AbstractVector{T}, v.data), v.pattern)
end

# Template instantiation: an all-zeros, `T`-valued vector over a given sparsity
# pattern (then fill the allowed entries with `setindex!`). The `{T}` is what
# marks the boolean vector as a *pattern* rather than data to infer nonzeros from.
FixedSparsityVector{T}(pattern::AbstractVector{Bool}) where {T} =
    FixedSparsityVector(zeros(T, size(pattern)), pattern)

# Disambiguate the exotic Bool-valued FixedSparsityVector (which is both a
# FixedSparsityVector and an AbstractVector{Bool}) in favour of element conversion.
FixedSparsityVector{T}(v::FixedSparsityVector{Bool}) where {T} =
    _withvec(convert(AbstractVector{T}, v.data), v.pattern)

# ---- Accessors ----

"""
    pattern(v::FixedSparsityVector) -> AbstractVector{Bool}

The pattern mask: `true` exactly at the positions allowed to be nonzero. Returns
the vector's live internal mask — treat it as read-only (mutating it would break
the fixed-pattern invariant).
"""
pattern(v::FixedSparsityVector) = v.pattern

"""
    parent(v::FixedSparsityVector)

The underlying data vector (with forbidden entries held at zero). Returns the
live internal storage — treat it as read-only.
"""
Base.parent(v::FixedSparsityVector) = v.data

# ---- AbstractArray interface ----

Base.size(v::FixedSparsityVector) = size(v.data)
Base.axes(v::FixedSparsityVector) = axes(v.data)
Base.IndexStyle(::Type{<:FixedSparsityVector}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(v::FixedSparsityVector, i::Int) = v.data[i]

Base.@propagate_inbounds function Base.setindex!(v::FixedSparsityVector, x, i::Int)
    if v.pattern[i]
        v.data[i] = x
    else
        iszero(x) || throw(ArgumentError(
            "cannot set entry (linear index $i) to a nonzero value; it is fixed to zero by the sparsity pattern"))
    end
    return x
end

# Generic algorithms that `similar` then fill every entry would hit forbidden
# positions, so `similar` deliberately degrades to a plain dense `Vector`.
Base.similar(::FixedSparsityVector, ::Type{Tv}, dims::Dims) where {Tv} = Vector{Tv}(undef, dims)

# Independent data, shared (read-only) pattern.
Base.copy(v::FixedSparsityVector) = _wrapvec(copy(v.data), v.pattern)
Base.zero(v::FixedSparsityVector) = _wrapvec(zero(v.data), v.pattern)

# ---- broadcasting ----
# Broadcasting reads through the underlying dense data, so `v .+ 1`, `f.(v)`,
# etc. produce ordinary arrays rather than (possibly pattern-violating) wrappers.
Base.broadcastable(v::FixedSparsityVector) = v.data

# ---- scaling and sign (pattern-preserving; pattern shared) ----

Base.:*(v::FixedSparsityVector, c::Number) = _wrapvec(v.data * c, v.pattern)
Base.:*(c::Number, v::FixedSparsityVector) = _wrapvec(c * v.data, v.pattern)
Base.:/(v::FixedSparsityVector, c::Number) = _wrapvec(v.data / c, v.pattern)
Base.:-(v::FixedSparsityVector) = _wrapvec(-v.data, v.pattern)

# ---- addition / subtraction ----
# Between two fixed-sparsity vectors the result pattern is the union of patterns.
function Base.:+(u::FixedSparsityVector, v::FixedSparsityVector)
    size(u) == size(v) || throw(DimensionMismatch("dimensions must match: $(size(u)) vs $(size(v))"))
    return _wrapvec(u.data + v.data, u.pattern .| v.pattern)
end
function Base.:-(u::FixedSparsityVector, v::FixedSparsityVector)
    size(u) == size(v) || throw(DimensionMismatch("dimensions must match: $(size(u)) vs $(size(v))"))
    return _wrapvec(u.data - v.data, u.pattern .| v.pattern)
end
# Against an unconstrained vector the result is unconstrained → dense.
Base.:+(u::FixedSparsityVector, v::AbstractVector) = u.data + v
Base.:+(u::AbstractVector, v::FixedSparsityVector) = u + v.data
Base.:-(u::FixedSparsityVector, v::AbstractVector) = u.data - v
Base.:-(u::AbstractVector, v::FixedSparsityVector) = u - v.data

# ---- transpose / adjoint / products / solves ----
# Intentionally NOT given bespoke methods. `transpose(v)` / `adjoint(v)` yield a
# *row* (a 1×n object), which is no longer a vector pattern, so the lazy
# `Transpose`/`Adjoint` wrappers from LinearAlgebra are the right result and read
# correctly through our `getindex`. Likewise matrix–vector products and solves
# fall back to the generic `AbstractVector` paths (our `similar` returns a
# `Vector`), producing dense results.
