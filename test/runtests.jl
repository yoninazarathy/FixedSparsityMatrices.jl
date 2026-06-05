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

    @testset "template: zero-valued matrix from a pattern" begin
        pat = Bool[1 1 0; 0 1 1; 1 0 1]
        A = FixedSparsityMatrix{Float64}(pat)
        @test eltype(A) == Float64
        @test all(iszero, A)
        @test pattern(A) == pat
        @test pattern(A) isa BitMatrix
        @test size(A) == (3, 3)
        # fill allowed entries; forbidden ones stay forbidden
        A[1, 1] = 5.0
        @test A[1, 1] == 5.0
        @test_throws ArgumentError (A[1, 3] = 2.0)
        # element type is honoured
        @test eltype(FixedSparsityMatrix{Int}(pat)) == Int
        # contrast: a *bare* Bool matrix is treated as data (pattern inferred from
        # its `true`s), giving a Bool-valued matrix — not a template.
        @test eltype(FixedSparsityMatrix(Bool[1 0; 0 1])) == Bool
    end

    @testset "element types beyond Float64" begin
        Ai = FixedSparsityMatrix([1 2; 0 3], Bool[1 1; 0 1])
        @test eltype(Ai) == Int
        @test Ai[2, 1] == 0
        @test 2Ai isa FixedSparsityMatrix
        @test Matrix(2Ai) == [2 4; 0 6]

        Ac = FixedSparsityMatrix(ComplexF64[1+1im 2; 0 3], Bool[1 1; 0 1])
        @test eltype(Ac) == ComplexF64
        @test Matrix(adjoint(Ac)) == adjoint(Matrix(Ac))      # conjugate transpose
        @test pattern(adjoint(Ac)) == permutedims(Bool[1 1; 0 1])
    end

    @testset "pattern sharing optimization" begin
        A = FixedSparsityMatrix([1.0 2.0; 0.0 3.0], Bool[1 1; 0 1])
        # pattern-preserving ops share the SAME pattern object (no copy)
        @test pattern(2A) === pattern(A)
        @test pattern(A * 2) === pattern(A)
        @test pattern(A / 2) === pattern(A)
        @test pattern(-A) === pattern(A)
        @test pattern(copy(A)) === pattern(A)
        @test pattern(zero(A)) === pattern(A)
        # data stays independent: mutating a derived matrix does not touch A
        B = copy(A)
        B[1, 1] = 99.0
        @test A[1, 1] == 1.0
        Z = zero(A)
        Z[1, 1] = 7.0
        @test A[1, 1] == 1.0
        # transpose/+ compute a fresh (correct) pattern rather than sharing
        @test pattern(transpose(A)) == permutedims(pattern(A))
        @test pattern(A + A) == pattern(A)
    end

    @testset "no method ambiguities" begin
        @test isempty(detect_ambiguities(FixedSparsityMatrices))
    end

    # ------------------------------------------------------------------
    # FixedSparsityVector
    # ------------------------------------------------------------------

    @testset "vector: explicit mask (construction zeroes forbidden entries)" begin
        data = [1.0, 9.0, 3.0]            # entry 2 is outside the pattern but nonzero
        pat = Bool[1, 0, 1]
        v = FixedSparsityVector(data, pat)
        dense = [1.0, 0.0, 3.0]
        @test Vector(v) == dense          # the 9.0 was forced to 0
        @test data[2] == 9.0              # original argument untouched (defensive copy)
        test_fixedsparsity(v, dense, pat)
    end

    @testset "vector: infer pattern from nonzeros" begin
        data = [1.0, 0.0, 2.0]
        v = FixedSparsityVector(data)
        @test pattern(v) == Bool[1, 0, 1]
        test_fixedsparsity(v, data, Bool[1, 0, 1])
    end

    @testset "vector: pattern is a BitVector by default" begin
        v = FixedSparsityVector([1.0, 0.0, 3.0], Bool[1, 0, 1])   # Vector{Bool} input
        @test pattern(v) isa BitVector
        @test FixedSparsityVector([1.0, 0.0]) |> pattern isa BitVector   # inferred
        # pattern-preserving ops keep the BitVector backend
        @test pattern(2v) isa BitVector
        @test pattern(copy(v)) isa BitVector
        # explicit opt-out via the parametric constructor keeps Vector{Bool}
        w = FixedSparsityVector{Float64, Vector{Float64}, Vector{Bool}}([1.0, 0.0], Bool[1, 0])
        @test pattern(w) isa Vector{Bool}
        @test pattern(2w) isa Vector{Bool}
        @test pattern(copy(w)) isa Vector{Bool}
    end

    @testset "vector: dimension mismatch" begin
        @test_throws DimensionMismatch FixedSparsityVector([1.0, 2.0], Bool[1, 0, 1])
    end

    @testset "vector: element-type conversion preserves pattern" begin
        v = FixedSparsityVector([1.0, 0.0, 2.0])
        v32 = FixedSparsityVector{Float32}(v)
        @test eltype(v32) == Float32
        @test pattern(v32) == pattern(v)
        @test Vector(v32) == Float32[1, 0, 2]
    end

    @testset "vector: addition unions the patterns" begin
        u = FixedSparsityVector([1.0, 2.0, 0.0], Bool[1, 1, 0])
        v = FixedSparsityVector([0.0, 0.0, 3.0], Bool[0, 0, 1])
        w = u + v
        @test w isa FixedSparsityVector
        @test pattern(w) == Bool[1, 1, 1]
        @test Vector(w) == [1.0, 2.0, 3.0]
        # against a plain vector it degrades to dense
        @test (u + [1.0, 1.0, 1.0]) isa Vector
    end

    @testset "vector: products / dot / matvec degrade to dense" begin
        v = FixedSparsityVector([1.0, 0.0, 3.0], Bool[1, 0, 1])
        @test dot(v, v) == 10.0
        @test v' * v == 10.0
        M = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        @test M * v == [1.0, 0.0, 3.0]
        @test (M * v) isa Vector
    end

    @testset "vector: template from a pattern" begin
        pat = Bool[1, 0, 1, 1]
        v = FixedSparsityVector{Float64}(pat)
        @test eltype(v) == Float64
        @test all(iszero, v)
        @test pattern(v) == pat
        @test pattern(v) isa BitVector
        @test length(v) == 4
        v[1] = 5.0
        @test v[1] == 5.0
        @test_throws ArgumentError (v[2] = 2.0)
        @test eltype(FixedSparsityVector{Int}(pat)) == Int
        # a *bare* Bool vector is treated as data (pattern inferred), not a template
        @test eltype(FixedSparsityVector(Bool[1, 0, 1])) == Bool
    end

    @testset "vector: pattern sharing optimization" begin
        v = FixedSparsityVector([1.0, 0.0, 3.0], Bool[1, 0, 1])
        @test pattern(2v) === pattern(v)
        @test pattern(v * 2) === pattern(v)
        @test pattern(v / 2) === pattern(v)
        @test pattern(-v) === pattern(v)
        @test pattern(copy(v)) === pattern(v)
        @test pattern(zero(v)) === pattern(v)
        # data stays independent
        w = copy(v); w[1] = 99.0
        @test v[1] == 1.0
        z = zero(v); z[1] = 7.0
        @test v[1] == 1.0
    end

    @testset "vector: in-place broadcast respects the pattern" begin
        v = FixedSparsityVector([1.0, 0.0, 3.0], Bool[1, 0, 1])
        c = copy(v)
        c .= [10.0, 0.0, 30.0]                  # zero at forbidden index 2: OK
        @test Vector(c) == [10.0, 0.0, 30.0]
        @test_throws ArgumentError (copy(v) .= [10.0, 7.0, 30.0])   # nonzero at index 2
        @test_throws ArgumentError (copy(v) .= 1.0)                 # scalar to all entries
        d = copy(v); d .= 0.0
        @test all(iszero, d)
        fill!(copy(v), 0.0)                      # zero fill always allowed
        @test_throws ArgumentError fill!(copy(v), 5.0)
    end

    @testset "vector: element types beyond Float64" begin
        vi = FixedSparsityVector([1, 0, 3], Bool[1, 0, 1])
        @test eltype(vi) == Int
        @test vi[2] == 0
        @test 2vi isa FixedSparsityVector
        @test Vector(2vi) == [2, 0, 6]

        vc = FixedSparsityVector(ComplexF64[1+1im, 0, 3], Bool[1, 0, 1])
        @test eltype(vc) == ComplexF64
        @test Vector(vc) == ComplexF64[1+1im, 0, 3]
    end

    @testset "shared abstract supertype" begin
        A = FixedSparsityMatrix([1.0 0.0; 0.0 2.0])
        v = FixedSparsityVector([1.0, 0.0])
        @test A isa AbstractFixedSparsityArray
        @test v isa AbstractFixedSparsityArray
        @test A isa AbstractMatrix      # still a genuine AbstractMatrix
        @test v isa AbstractVector      # still a genuine AbstractVector
    end

end
