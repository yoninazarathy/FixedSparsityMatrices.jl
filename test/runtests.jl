using FixedSparsityMatrices
using LinearAlgebra
using SparseArrays
using Test

include("testutils.jl")

@testset "FixedSparsityMatrices.jl" begin

    @testset "explicit mask (construction zeroes forbidden entries)" begin
        # (1,3) is outside the support but nonzero in `data` — it must be zeroed.
        data = [1.0 2.0 9.0;
                0.0 3.0 4.0;
                5.0 0.0 6.0]
        support = Bool[1 1 0; 0 1 1; 1 0 1]
        A = FixedSparsityMatrix(data, support)
        dense = [1.0 2.0 0.0;
                 0.0 3.0 4.0;
                 5.0 0.0 6.0]
        @test Matrix(A) == dense          # the 9.0 was forced to 0
        @test data[1, 3] == 9.0           # original argument untouched (defensive copy)
        test_fixedsparsity(A, dense, support)
    end

    @testset "infer support from nonzeros" begin
        data = [1.0 0.0; 0.0 2.0]
        A = FixedSparsityMatrix(data)
        @test pattern(A) == Bool[1 0; 0 1]
        test_fixedsparsity(A, data, Bool[1 0; 0 1])
    end

    @testset "interop: LinearAlgebra shape families" begin
        D = Diagonal([1.0, 2.0, 3.0])
        test_fixedsparsity(FixedSparsityMatrix(D), Matrix(D), Bool[1 0 0; 0 1 0; 0 0 1])

        BU = Bidiagonal([1.0, 2.0, 3.0], [4.0, 5.0], :U)
        test_fixedsparsity(FixedSparsityMatrix(BU), Matrix(BU), Bool[1 1 0; 0 1 1; 0 0 1])

        BL = Bidiagonal([1.0, 2.0, 3.0], [4.0, 5.0], :L)
        test_fixedsparsity(FixedSparsityMatrix(BL), Matrix(BL), Bool[1 0 0; 1 1 0; 0 1 1])

        Tri = Tridiagonal([1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0])
        test_fixedsparsity(FixedSparsityMatrix(Tri), Matrix(Tri), Bool[1 1 0; 1 1 1; 0 1 1])

        U = UpperTriangular([1.0 2.0 3.0; 0.0 4.0 5.0; 0.0 0.0 6.0])
        test_fixedsparsity(FixedSparsityMatrix(U), Matrix(U), Bool[1 1 1; 0 1 1; 0 0 1])

        # A structural band entry that is numerically zero stays *in* the support.
        BUz = Bidiagonal([1.0, 2.0, 3.0], [0.0, 5.0], :U)   # super-diagonal (1,2) is 0
        Afsz = FixedSparsityMatrix(BUz)
        @test pattern(Afsz)[1, 2] == true                   # still allowed to be nonzero
        @test Afsz[1, 2] == 0.0
    end

    @testset "interop: sparse stored pattern" begin
        S = sparse([1, 2, 3], [1, 3, 2], [1.0, 2.0, 3.0], 3, 3)
        A = FixedSparsityMatrix(S)
        @test pattern(A) == Bool[1 0 0; 0 0 1; 0 1 0]
        test_fixedsparsity(A, Matrix(S), Bool[1 0 0; 0 0 1; 0 1 0])
    end

    @testset "dimension mismatch" begin
        @test_throws DimensionMismatch FixedSparsityMatrix([1.0 2.0], Bool[1 1; 0 0])
    end

    @testset "element-type conversion preserves pattern" begin
        A = FixedSparsityMatrix([1.0 0.0; 0.0 2.0])
        A32 = FixedSparsityMatrix{Float32}(A)
        @test eltype(A32) == Float32
        @test pattern(A32) == pattern(A)
        @test Matrix(A32) == Float32[1 0; 0 2]
    end

    @testset "addition unions the supports" begin
        A = FixedSparsityMatrix([1.0 2.0; 0.0 0.0], Bool[1 1; 0 0])
        B = FixedSparsityMatrix([0.0 0.0; 3.0 4.0], Bool[0 0; 1 1])
        C = A + B
        @test C isa FixedSparsityMatrix
        @test pattern(C) == Bool[1 1; 1 1]
        @test Matrix(C) == [1.0 2.0; 3.0 4.0]
        # against a plain matrix it degrades to dense
        @test (A + [1.0 1.0; 1.0 1.0]) isa Matrix
    end

    @testset "products degrade to dense" begin
        A = FixedSparsityMatrix([1.0 2.0; 0.0 3.0], Bool[1 1; 0 1])
        x = [1.0, 1.0]
        @test A * x == [3.0, 3.0]
        @test (A * A) isa Matrix
        @test A * A == Matrix(A) * Matrix(A)
        @test A \ x == Matrix(A) \ x
    end

end
