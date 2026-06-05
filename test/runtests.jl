using FixedSparsityMatrices
using LinearAlgebra
using SparseArrays
using Test

include("testutils.jl")

@testset "FixedSparsityMatrices.jl" begin

    @testset "explicit mask (construction zeroes forbidden entries)" begin
        # (1,3) is outside the pattern but nonzero in `data` — it must be zeroed.
        data = [1.0 2.0 9.0;
                0.0 3.0 4.0;
                5.0 0.0 6.0]
        pat = Bool[1 1 0; 0 1 1; 1 0 1]
        A = FixedSparsityMatrix(data, pat)
        dense = [1.0 2.0 0.0;
                 0.0 3.0 4.0;
                 5.0 0.0 6.0]
        @test Matrix(A) == dense          # the 9.0 was forced to 0
        @test data[1, 3] == 9.0           # original argument untouched (defensive copy)
        test_fixedsparsity(A, dense, pat)
    end

    @testset "infer pattern from nonzeros" begin
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

        # A structural band entry that is numerically zero stays *in* the pattern.
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

    @testset "pattern is stored as a BitMatrix by default" begin
        A = FixedSparsityMatrix([1.0 2.0; 0.0 3.0], Bool[1 1; 0 1])   # Matrix{Bool} input
        @test pattern(A) isa BitMatrix
        @test FixedSparsityMatrix([1.0 0.0; 0.0 2.0]) |> pattern isa BitMatrix   # inferred
        @test FixedSparsityMatrix(Diagonal([1.0, 2.0])) |> pattern isa BitMatrix # structural
        # pattern-preserving ops keep the BitMatrix backend
        @test pattern(2A) isa BitMatrix
        @test pattern(transpose(A)) isa BitMatrix
        @test pattern(copy(A)) isa BitMatrix
        # explicit opt-out via the parametric constructor keeps Matrix{Bool}
        B = FixedSparsityMatrix{Float64, Matrix{Float64}, Matrix{Bool}}([1.0 2.0; 0.0 3.0], Bool[1 1; 0 1])
        @test pattern(B) isa Matrix{Bool}
        @test pattern(2B) isa Matrix{Bool}        # ops preserve the chosen backend
        @test pattern(copy(B)) isa Matrix{Bool}
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

    @testset "addition unions the patterns" begin
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

    @testset "non-square matrices" begin
        data = [1.0 2.0 0.0; 0.0 3.0 4.0]
        pat = Bool[1 1 0; 0 1 1]
        A = FixedSparsityMatrix(data, pat)
        @test size(A) == (2, 3)
        test_fixedsparsity(A, data, pat)
    end

    @testset "fill! and in-place broadcast respect the pattern" begin
        A = FixedSparsityMatrix([1.0 2.0; 0.0 3.0], Bool[1 1; 0 1])

        B = copy(A); fill!(B, 0.0)                 # zero fill always allowed
        @test all(iszero, B)
        @test pattern(B) == pattern(A)
        @test_throws ArgumentError fill!(copy(A), 5.0)   # nonzero hits forbidden (2,1)

        C = copy(A)
        C .= [10.0 20.0; 0.0 30.0]                 # zero at forbidden (2,1): OK
        @test Matrix(C) == [10.0 20.0; 0.0 30.0]
        @test_throws ArgumentError (copy(A) .= [10.0 20.0; 7.0 30.0])  # nonzero at (2,1)
        @test_throws ArgumentError (copy(A) .= 1.0)                    # scalar to all entries
        D = copy(A); D .= 0.0
        @test all(iszero, D)
    end

    @testset "random instance honoring a pattern (idiom)" begin
        pat = Bool[1 1 0; 0 1 1; 1 0 1]
        R = FixedSparsityMatrix(rand(3, 3), pat)
        @test pattern(R) == pat
        for I in CartesianIndices(R)
            pat[I] || @test R[I] == 0.0
        end
    end

end
