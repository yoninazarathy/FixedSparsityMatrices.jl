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
