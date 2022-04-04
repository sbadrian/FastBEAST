include("test_common.jl")

@testset "FastBEAST.jl" begin
    include("test_boundingbox.jl")
    include("test_boxtree.jl")
    include("test_kmeans.jl")
    include("test_skeletons.jl")
    include("test_aca.jl")
    include("test_aca_beast.jl")
    include("test_hmatrix.jl")
    include("test_beast.jl")
    include("test_beast_efie.jl")
    include("test_beast_mfie.jl")
end