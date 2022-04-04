using ParallelKMeans

"""
    KMeansTreeOptions <: TreeOptions

Is the datatype that describes which tree the [`create_tree`](@ref) function creates.

# Fields
- `iterations`: number of iterations on each level, default is one iterations
- `nchildren`: defines the number of children of each node, default is two
- `nmin`: defines the minimum amount of datapoints which are needed in a 
    cluster so that it gets split up in subclusters, default is 1
- `maxlevel`: defines the maximum amount of levels, default is 100.
- `algorithm`: defines which algorithm is used. The :naive approach is not recommended.
    Default is the wrapped ParallelKMeans algorithm.
"""
struct KMeansTreeOptions <: TreeOptions
    iterations
    nchildren
    nmin
    maxlevel
    algorithm
end

function KMeansTreeOptions(;
    iterations=20,
    nchildren=2,
    nmin=1,
    maxlevel=100,
    algorithm=:ParallelKMeans
)
    return KMeansTreeOptions(iterations, nchildren, nmin, maxlevel, algorithm)
end

"""
    KMeansTreeNode{T} <: AbstractNode

Is the datatype of each node in the [K-Means Clustering Tree](@ref).

# Fields
- `parent::Union{KMeansTreeNode{T},Nothing}`: is the superordinate cluster of
    of the represented cluster
- `children::Union{Vector{KMeansTreeNode{T}}, Nothing}`: all directly 
    subordinated clusters of the represented cluster
- `level::Integer`: the level of the represented cluster
- `center`: the center by which the cluster is defined
- `radius`: the euclidian distance between the center and the farthest away
    point
- `data::T`: array containig the indices of the points in this cluster
"""
mutable struct KMeansTreeNode{D, I, F, N <: NodeData{I, F}} <: AbstractNode{I, F, N}
    parent::Union{KMeansTreeNode{D, I, F, N}, Nothing}
    children::Vector{KMeansTreeNode{D, I, F, N}}
    level::I
    center::SVector{D, F}
    radius::F
    data::N
end

function KMeansTreeNode(
    center::SVector{D, F},
    radius::F,
    data::N
) where {D, I, F, N <: NodeData{I, F}}

    return KMeansTreeNode(nothing, KMeansTreeNode{D, I, F, N}[], I(0), center, radius, data)
end

function add_child!(
    parent::KMeansTreeNode{D, I, F, N},
    center::SVector{D, F},
    radius::F,
    data::N
) where {D, I, F, N}
    
    childnode = KMeansTreeNode(
        parent,
        KMeansTreeNode{D, I, F, N}[],
        level(parent) + 1,
        center,
        radius,
        data
    )

    push!(parent.children, childnode)

end

"""
    create_tree(points::Array{SVector{D, T}, 1}; treeoptions)

Creates an algebraic tree for an given set of datapoints. The returned 
datastructure is the foundation for the algorithms in this package. 

# Arguments
- `points::Array{SVector{D, T}, 1}`: is an array of 
    [SVector](https://juliaarrays.github.io/StaticArrays.jl/stable/pages/api/#SVector-1). 
    Each 
    [SVector](https://juliaarrays.github.io/StaticArrays.jl/stable/pages/api/#SVector-1)
    contains in general two or three float values, which describe the position 
    in the space.

# Keywords
- `treeoptions::TreeOptions`: this keyword defines by which tree is build. 
    `TreeOptions` is an abstract type which either can be `BoxTreeOptions` or
    [`KMeansTreeOptions`](@ref). Default type is `BoxTreeOptions`.

# Returns
- `AbstractNode`: the root of the created tree. AbstractNode is an abstract type 
    which either can be `BoxTreeNode` or [`KMeansTreeNode`](@ref), depending on the keyword.
"""
function create_tree(
    points::Vector{SVector{D, F}},
    treeoptions::KMeansTreeOptions
) where {D, F}
    root = KMeansTreeNode(zeros(SVector{D, F}), F(0.0), ACAData(Vector(1:length(points)), F))
    if treeoptions.algorithm != :naive
        pointsM = reshape(
            [points[j][i] for j = 1:length(points) for i = 1:D], 
            (D,length(points))
        )
        fill_tree!(
            root,
            pointsM,
            nmin=treeoptions.nmin,
            maxlevel=treeoptions.maxlevel,
            iterations=treeoptions.iterations,
            nchildren=treeoptions.nchildren
        )
    else
        fill_tree!(
            root,
            points,
            nmin=treeoptions.nmin,
            maxlevel=treeoptions.maxlevel,
            iterations=treeoptions.iterations,
            nchildren=treeoptions.nchildren
        )
    end

    return root
end

function whichcenter(center, point, nchildren)
    centerindex = argmin(norm.([center[i]-point for i = 1:nchildren]))

    return centerindex
end

function fill_tree!(
    node::KMeansTreeNode{D, I, F, N},
    points::Vector{SVector{D, F}};
    nmin=1,
    maxlevel=log2(eps(eltype(points))),
    iterations, 
    nchildren
) where {D, I, F, N <: NodeData{I, F}}

    if numindices(node) <= nmin|| level(node) >= maxlevel - 1 || numindices(node) <= nchildren
        return
    end

    center = points[indices(node)[1:nchildren]]
    sorted_points = zeros(I, numindices(node)+1, nchildren)

    for _ = 1:iterations
        sorted_points .= 0
        for pindex in indices(node)
            id = whichcenter(center, points[pindex], nchildren)
            sorted_points[1, id] +=  1
            sorted_points[1 + sorted_points[1, id], id] = pindex
        end
        for k = 1:nchildren
            center[k] = sum(
                points[sorted_points[2:(sorted_points[1, k]+1), k]]
            ) ./ sorted_points[1, k]          
        end
    end
    

    for i = 1:nchildren
        if sorted_points[1, i] > 0
            childpointindices = sorted_points[2:(sorted_points[1, i]+1), i]
            radius = maximum(norm.([points[childpointindices[j]].-center[i] for j = 1:length(childpointindices)]))
            add_child!(node, center[i], radius, ACAData(childpointindices, F))
            fill_tree!(
                lastchild(node), 
                points, 
                nmin=nmin, 
                maxlevel=maxlevel, 
                iterations=iterations, 
                nchildren=nchildren
            )
        end
    end
end

function fill_tree!(
    node::KMeansTreeNode{D, I, F, N},
    pointsM::Matrix;
    nmin=1,
    maxlevel=log2(eps(eltype(points))),
    iterations, 
    nchildren
) where {D, I, F, N <: NodeData{I, F}}

    if numindices(node) <= nmin || level(node) >= maxlevel - 1 || numindices(node) <= nchildren
        return
    end
    
    sorted_points = zeros(I, length(indices(node))+1, nchildren)
    
    # A first (probably too naive?) heuristic
    opt_nthreads = max(1, floor(I, log10(0.001*length(indices(node)))))
    
    kmcluster = ParallelKMeans.kmeans(
        pointsM[:,indices(node)],
        nchildren,
        max_iters=iterations,
        n_threads=opt_nthreads
    )

    for (index, value) in enumerate(kmcluster.assignments)
        sorted_points[1, value] += 1
        sorted_points[sorted_points[1,value]+1, value] = indices(node)[index] 
    end

    for i = 1:nchildren
        if sorted_points[1, i] > 0
            center = SVector{D, F}([kmcluster.centers[j, i] for j = 1:D])
            radius = maximum(norm.(eachcol(
                pointsM[:, sorted_points[2:(sorted_points[1, i]+1),i]].-kmcluster.centers[:,i]
            )))
            
            add_child!(
                node,
                center,
                radius,
                ACAData(sorted_points[2:(sorted_points[1, i]+1), i], F)
            )
            fill_tree!(
                lastchild(node), 
                pointsM, 
                nmin=nmin, 
                maxlevel=maxlevel, 
                iterations=iterations, 
                nchildren=nchildren
            )
        end
    end
end

"""
    iscompressable(sourcenode::AbstractNode, testnode::AbstractNode)

Determines whether two nodes of a tree are comressable. The criteria differs 
between the [Box Tree](@ref) and the [K-Means Clustering Tree](@ref).
For the [K-Means Clustering Tree](@ref) two nodes can be compressed, if the distance 
between the centers of two clusters is greater than the sum of their radius 
multiplied by a factor of 1.5.
For the [Box Tree](@ref) two nodes can be compressed, if the distance between the centers 
of the two boxes is greater than the sum of the distances of each box's center to 
one of its corners multiplied by a factor of 1.1.

# Arguments
- `sourcenode::AbstractNode`: the node which is observed
- `testnode::AbstractNode`: the node which is tested for compression

# Returns
- `true`: if the input nodes are compressable
- `false`: if the input nodes are not compressable
"""
function iscompressable(sourcenode::KMeansTreeNode, testnode::KMeansTreeNode)
    if level(sourcenode) > 0 && level(testnode) > 0
        dist = norm(sourcenode.center - testnode.center)
        factor = 1.5
        if factor * (sourcenode.radius + testnode.radius) < dist
            return true
        else
            return false
        end
    else
        return false
    end
end