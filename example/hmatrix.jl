using FastBEAST
using StaticArrays
using LinearAlgebra
using Plots
using Printf
plotlyjs()

##
function logkernel(testpoint::SVector{2,T}, sourcepoint::SVector{2,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return - 2*π*log(norm(testpoint - sourcepoint))
    end
end

N =  1000
NT = 100

spoints = [@SVector rand(2) for i = 1:N]

tpoints = 0.1*[@SVector rand(2) for i = 1:NT] + [SVector(3.5, 3.5) for i = 1:NT]

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(
        promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
        length(testpoints), 
        length(sourcepoints)
    )

    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end

function assembler(kernel, matrix, testpoints, sourcepoints)
    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

kmat = assembler(logkernel, tpoints, spoints)

U, S, V = svd(kmat)

println("Condition number: ", S[1] / S[end])

plot(S, yaxis=:log, marker=:x)

##
logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel,
    matrix,
    tpoints[tdata],
    spoints[sdata]
)
stree = create_tree(spoints, BoxTreeOptions(nmin=100))
ttree = create_tree(tpoints, BoxTreeOptions(nmin=100))
kmat = assembler(logkernel, tpoints, spoints)
hmat = HMatrix(logkernelassembler, ttree, stree, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat, kmat))

v2 = rand(NT)
@printf("Accuracy test: %.2e\n", norm(adjoint(hmat)*v2 - 
    adjoint(kmat)*v2)/norm(adjoint(kmat)*v2))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

#Comparison KMeans/BoxTree
## Example1 Linear function
N = 10000
spoints = [SVector(i,i) for i = 1:N] + [SVector(1.0, 1.0) for i = 1:N]

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=100))
kmat = assembler(logkernel, spoints, spoints)
@time hmat = HMatrix(logkernelassembler, stree, stree, compressor=:naive, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100,nchildren=2,nmin=20))
kmat = assembler(logkernel, spoints, spoints)
@time hmat = HMatrix(logkernelassembler, stree, stree, compressor=:naive, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

## Example2 Random distribution
N = 2000
spoints = [@SVector rand(2) for i = 1:N]

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel, 
    matrix, 
    spoints[tdata], 
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=100))
kmat = assembler(logkernel, spoints, spoints)
@time hmat = HMatrix(logkernelassembler, stree, stree, compressor=:naive, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100,nchildren=2,nmin=20))
kmat = assembler(logkernel, spoints, spoints)
@time hmat = HMatrix(logkernelassembler, stree, stree, compressor=:naive, Int64, Float64,)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

## Example3 Defined target points
N =  10000
NT = 1000

spoints = [@SVector rand(2) for i = 1:N]
tpoints = 0.1*[@SVector rand(2) for i = 1:NT] + [SVector(1.0, 1.0) for i = 1:NT]

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel,
    matrix,
    tpoints[tdata],
    spoints[sdata]
)

ttree = create_tree(tpoints, BoxTreeOptions(nmin=10))
stree = create_tree(spoints, BoxTreeOptions(nmin=10))
kmat = assembler(logkernel, tpoints, spoints)
@time hmat = HMatrix(logkernelassembler, ttree, stree, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat, kmat))

v2 = rand(NT)
@printf(
    "Accuracy test: %.2e\n",
    norm(adjoint(hmat)*v2 - adjoint(kmat)*v2)/norm(adjoint(kmat)*v2)
)
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100, nchildren=2, nmin=5))
ttree = create_tree(tpoints, KMeansTreeOptions(iterations=100, nchildren=2, nmin=5))
kmat = assembler(logkernel, tpoints, spoints)
@time hmat = HMatrix(logkernelassembler, ttree, stree, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat, kmat))

v2 = rand(NT)
@printf(
    "Accuracy test: %.2e\n", 
    norm(adjoint(hmat)*v2 - adjoint(kmat)*v2)/norm(adjoint(kmat)*v2)
)
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)