struct LazyMatrix{I, F} <: AbstractMatrix{F}
    μ::Function
    τ::Vector{I}
    σ::Vector{I}
end

Base.size(A::LazyMatrix) = (length(A.τ), length(A.σ))

function Base.getindex(
    A::T,
    I,
    J
) where {K, F, T <: LazyMatrix{K, F}}

    Z = zeros(F, length(I), length(J))
    A.μ(Z, view(A.τ, I), view(A.σ, J))
    return Z
end

function LazyMatrix(μ::Function, τ::Vector{I}, σ::Vector{I}, ::Type{F}) where {I, F}
    return LazyMatrix{I, F}(μ, τ, σ)
end

@views function (A::LazyMatrix{K, F})(Z::S, I, J) where {K, F, S <: AbstractMatrix{F}}
    A.μ(view(Z, I, J), view(A.τ, I), view(A.σ, J))
end

struct ACAGlobalMemory{F}
    U::Matrix{F}
    V::Matrix{F}
    used_I::Vector{Bool}
    updated_I::Vector{Int}
    used_J::Vector{Bool}
end

function allocate_aca_memory(::Type{F}, maxrows, maxcolumns; maxrank = 40) where {F}
    U = zeros(F, maxrows, maxrank)
    V = zeros(F, maxrank, maxcolumns)
    used_I = zeros(Bool, maxrows)
    updated_I = zeros(Int, maxrows)
    used_J = zeros(Bool, maxcolumns)
    return ACAGlobalMemory(U, V, used_I, updated_I, used_J)
end

maxrank(acamemory::ACAGlobalMemory) = size(acamemory.U, 2)


function smartmaxlocal(roworcolumn, acausedindices)
    maxval = -1
    index = -1
    for i=1:length(roworcolumn)
        if !acausedindices[i]
            if abs(roworcolumn[i]) > maxval
                maxval = abs(roworcolumn[i]) 
                index = i
            end
        end
    end

    return index, maxval
end

function aca(
    M::LazyMatrix{I, F},
    am::ACAGlobalMemory{F};
    tol=1e-14,
    isblockstructured=false,
    svdrecompress=true
) where {I, F}

    Ic = 1
    Jc = 1

    (maxrows, maxcolumns) = size(M)

    nextrow = 1
    am.used_I[nextrow] = true

    i = 1

    @views M.μ(
        am.V[Ic:Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )
    while isapprox(norm(am.V[Ic:Ic, 1:maxcolumns]), 0.0) && nextrow != maxrows
        i += 1
        nextrow += 1
        am.used_I[nextrow] = true
        @views M.μ(
            am.V[Ic:Ic, 1:maxcolumns], 
            M.τ[nextrow:nextrow],
            M.σ[1:size(M,2)]
        )
    end

    @views nextcolumn, maxval = smartmaxlocal(
        am.V[Ic:Ic, 1:maxcolumns],
        am.used_J[1:maxcolumns]
    )

    am.used_J[nextcolumn] = true

    dividor = am.V[Ic, nextcolumn]
    
    if !isapprox(maximum(dividor), 0)

        @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor
        @views M.μ(
            am.U[1:maxrows, Jc:Jc], 
            M.τ[1:size(M, 1)], 
            M.σ[nextcolumn:nextcolumn]
        )

        @views normUVlastupdate = norm(am.U[1:maxrows, 1])*norm(am.V[1, 1:maxcolumns])
        normUVsqared = normUVlastupdate^2
        while i <= length(M.τ)-1 &&
            i <= length(M.σ)-1 &&
            Jc < maxrank(am)
            
            if isblockstructured
                for k = 1:length(M.τ)
                    if !isapprox(0, am.U[k,Ic]; atol=10^-16)
                        am.updated_I[k] += 1;
                    end
                end
            end

            if normUVlastupdate < sqrt(normUVsqared)*tol
                if isblockstructured
                    missingrow = 0
                    for k = 1:length(M.τ)
                        if am.updated_I[k] == 0
                            missingrow = k
                            break
                        end
                    end
                    if missingrow != 0
                        i += 1
                        maxval = am.U[missingrow,Jc]
                        nextrow = missingrow
                    else
                        break
                    end
                else
                    break
                end
            else
                i += 1
                @views nextrow, maxval = smartmaxlocal(
                    am.U[1:maxrows,Jc],
                    am.used_I[1:maxrows]
                )
            end
    
            am.used_I[nextrow] = true

            Ic += 1
            @views M.μ(
                am.V[Ic:Ic, 1:maxcolumns],
                M.τ[nextrow:nextrow],
                M.σ[1:size(M, 2)]
            )

            while isapprox(norm(am.V[Ic:Ic, 1:maxcolumns]), 0.0) && nextrow != maxrows
                nextrow += 1
                i+=1
                am.used_I[nextrow] = true
                @views M.μ(
                    am.V[Ic:Ic, 1:maxcolumns], 
                    M.τ[nextrow:nextrow],
                    M.σ[1:size(M, 2)]
                )
            end

            @assert Jc == (Ic - 1)
            for k = 1:Jc
                for kk=1:maxcolumns
                    am.V[Ic, kk] -= am.U[nextrow, k]*am.V[k, kk]
                end
            end

            @views nextcolumn, maxval = smartmaxlocal(
                am.V[Ic, 1:maxcolumns],
                am.used_J[1:maxcolumns]
            )

            if (isapprox(am.V[Ic, nextcolumn], 0.0))
                normUVlastupdate = 0.0
                am.V[Ic:Ic, 1:maxcolumns] .= 0.0
                Ic -= 1
                println("Matrix seems to have exact rank: ", Ic)
            else
                dividor = am.V[Ic, nextcolumn]
                @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor

                am.used_J[nextcolumn] = true

                Jc += 1
                @views M.μ(
                    am.U[1:maxrows, Jc:Jc], 
                    M.τ[1:size(M, 1)],
                    M.σ[nextcolumn:nextcolumn]
                )

                @assert Jc == Ic
                for k = 1:(Jc-1)
                    for kk = 1:maxrows
                        am.U[kk, Jc] -= am.U[kk, k]*am.V[k, nextcolumn]
                    end
                end

                @views normUVlastupdate = norm(am.U[1:maxrows,Jc])*norm(am.V[Ic,1:maxcolumns])

                normUVsqared += normUVlastupdate^2
                for j = 1:(Jc-1)
                    @views normUVsqared += 2*abs(dot(
                        am.U[1:maxrows,Jc],
                        am.U[1:maxrows,j])*dot(am.V[Ic,1:maxcolumns],
                        am.V[j,1:maxcolumns]
                    ))
                end
            end
        end

        if Jc == maxrank(am)
            println("WARNING: aborted ACA after maximum allowed rank")
        end
    end
    
    if svdrecompress && Jc > 1
        @views Q,R = qr(am.U[1:maxrows,1:Jc])
        @views U,s,V = svd(R*am.V[1:Ic,1:maxcolumns])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[1:maxrows,1:opt_r]
        B = (diagm(s)*V')[1:opt_r,1:maxcolumns]

        am.U[1:maxrows, 1:Jc] .= 0.0
        am.V[1:Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false

        return A, B
    else
        retU = am.U[1:maxrows,1:Jc]
        retV = am.V[1:Ic,1:maxcolumns]
        am.U[1:maxrows, 1:Jc] .= 0.0
        am.V[1:Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false

        return retU, retV
    end    
end

function aca(
    M::LazyMatrix{I, F};
    tol=1e-14,
    maxrank=40,
    isblockstructured=false,
    svdrecompress=true
) where {I, F}

    return aca(
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        tol=tol,
        isblockstructured=isblockstructured,
        svdrecompress=svdrecompress
    )
end
