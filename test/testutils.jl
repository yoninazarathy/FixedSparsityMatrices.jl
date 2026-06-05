# Reusable driver that exercises the full AbstractMatrix interface and the
# fixed-sparsity semantics of a `FixedSparsityMatrix`.
#
#   A     the matrix under test
#   dense the expected dense values (forbidden entries already zeroed)
#   pat   the expected boolean pattern mask

using FixedSparsityMatrices, LinearAlgebra, SparseArrays, Test

function test_fixedsparsity(A::FixedSparsityMatrix, dense::AbstractMatrix, pat::AbstractMatrix{Bool})
    @testset "interface: $(size(A,1))×$(size(A,2)) $(typeof(A).name.name)" begin
        # shape
        @test size(A) == size(dense)
        @test axes(A) == axes(dense)
        @test eltype(A) == eltype(dense)

        # pattern + forbidden entries are exactly zero
        @test pattern(A) == pat
        @test parent(A) === A.data
        for I in CartesianIndices(A)
            @test A[I] == dense[I]
            pat[I] || @test A[I] == zero(eltype(A))
        end

        # conversions
        @test Matrix(A) == dense
        @test collect(A) == dense
        @test diag(A) == diag(dense)
        @test Matrix(sparse(A)) == dense

        # scaling / sign preserve type, pattern and values
        for B in (2A, A * 2, A / 2, -A)
            @test B isa FixedSparsityMatrix
            @test pattern(B) == pat
        end
        @test Matrix(2A) == 2dense
        @test Matrix(A / 2) == dense / 2
        @test Matrix(-A) == -dense

        # transpose preserves the (transposed) pattern and values
        At = transpose(A)
        @test At isa FixedSparsityMatrix
        @test pattern(At) == permutedims(pat)
        @test Matrix(At) == permutedims(dense)

        # similar degrades to a plain dense Matrix (so generic code can fill it)
        s = similar(A)
        @test s isa Matrix
        @test size(s) == size(A)

        # broadcasting degrades to a dense array
        b = A .+ 1
        @test b isa Array
        @test b == dense .+ 1

        # setindex! semantics (operate on a copy)
        C = copy(A)
        @test C isa FixedSparsityMatrix
        idx = findfirst(pat)
        if idx !== nothing
            C[idx] = 42
            @test C[idx] == 42
        end
        zidx = findfirst(.!pat)
        if zidx !== nothing
            @test (C[zidx] = 0) == 0        # setting a forbidden entry to zero is a no-op
            @test C[zidx] == 0
            @test_throws ArgumentError C[zidx] = 1   # ...but nonzero is forbidden
        end
    end
    return nothing
end

# Vector analogue of the driver above.
#
#   v     the vector under test
#   dense the expected dense values (forbidden entries already zeroed)
#   pat   the expected boolean pattern mask
function test_fixedsparsity(v::FixedSparsityVector, dense::AbstractVector, pat::AbstractVector{Bool})
    @testset "interface: length-$(length(v)) $(typeof(v).name.name)" begin
        # shape
        @test size(v) == size(dense)
        @test length(v) == length(dense)
        @test axes(v) == axes(dense)
        @test eltype(v) == eltype(dense)

        # pattern + forbidden entries are exactly zero
        @test pattern(v) == pat
        @test parent(v) === v.data
        for i in eachindex(v)
            @test v[i] == dense[i]
            pat[i] || @test v[i] == zero(eltype(v))
        end

        # conversions
        @test Vector(v) == dense
        @test collect(v) == dense
        @test Vector(sparse(v)) == dense

        # scaling / sign preserve type, pattern and values
        for w in (2v, v * 2, v / 2, -v)
            @test w isa FixedSparsityVector
            @test pattern(w) == pat
        end
        @test Vector(2v) == 2dense
        @test Vector(v / 2) == dense / 2
        @test Vector(-v) == -dense

        # similar degrades to a plain dense Vector (so generic code can fill it)
        s = similar(v)
        @test s isa Vector
        @test size(s) == size(v)

        # broadcasting degrades to a dense array
        b = v .+ 1
        @test b isa Array
        @test b == dense .+ 1

        # setindex! semantics (operate on a copy)
        C = copy(v)
        @test C isa FixedSparsityVector
        idx = findfirst(pat)
        if idx !== nothing
            C[idx] = 42
            @test C[idx] == 42
        end
        zidx = findfirst(.!pat)
        if zidx !== nothing
            @test (C[zidx] = 0) == 0        # setting a forbidden entry to zero is a no-op
            @test C[zidx] == 0
            @test_throws ArgumentError C[zidx] = 1   # ...but nonzero is forbidden
        end
    end
    return nothing
end
