using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers


function farquaddata(op::BEAST.MaxwellOperator3D,
    test_local_space::BEAST.RefSpace, trial_local_space::BEAST.RefSpace,
    test_charts, trial_charts)

    a, b = 0.0, 1.0
    # CommonVertex, CommonEdge, CommonFace rules

    tqd = quadpoints(test_local_space, test_charts, (1,6))
    bqd = quadpoints(trial_local_space, trial_charts, (1,7))
    leg = (BEAST._legendre(3,a,b), BEAST._legendre(4,a,b), BEAST._legendre(5,a,b),)

    # High accuracy rules (use them e.g. in LF MFIE scenarios)
    # tqd = quadpoints(test_local_space, test_charts, (8,8))
    # bqd = quadpoints(trial_local_space, trial_charts, (8,9))
    # leg = (_legendre(8,a,b), _legendre(10,a,b), _legendre(5,a,b),)


    return (tpoints=tqd, bpoints=bqd, gausslegendre=leg)
end


c = 3e8
μ = 4*π*1e-7
ε = 1/(μ*c^2)
f = 1e8
λ = c/f
k = 2*π/λ
ω = k*c
η = sqrt(μ/ε)

a = 1
Γ_orig = CompScienceMeshes.meshcuboid(a,a,a,0.05)
Γ = translate(Γ_orig,SVector(-a/2,-a/2,-a/2))

Φ, Θ = [0.0], range(0,stop=π,length=100)
pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]

# This is an electric dipole
# The pre-factor (1/ε) is used to resemble 
# (9.18) in Jackson's Classical Electrodynamics
E = (1/ε) * dipolemw3d(location=SVector(0.4,0.2,0), 
                    orientation=1e-9.*SVector(0.5,0.5,0), 
                    wavenumber=k)

n = BEAST.NormalVector()

𝒆 = (n × E) × n
H = (-1/(im*μ*ω))*curl(E)
𝒉 = (n × H) × n

𝓣 = Maxwell3D.singlelayer(wavenumber=k)
𝓝 = BEAST.NCross()
𝓚 = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γ)

println("Number of RWG functions: ", numfunctions(X))

T = hassemble(𝓣,X,X, 
                nmin=100, 
                threading=:multi, 
                farquaddata=farquaddata, 
                verbose=true, 
                svdrecompress=false)
e = assemble(𝒆,X)
##
println("Enter iterative solver")
@time j_EFIE, ch = IterativeSolvers.gmres(T, e, log=true, reltol=1e-4, maxiter=500)
println("Finished iterative solver part. Number of iterations: ", ch.iters)

nf_E_EFIE = potential(MWSingleLayerField3D(wavenumber=k), pts, j_EFIE, X)
nf_H_EFIE = potential(BEAST.MWDoubleLayerField3D(wavenumber=k), pts, j_EFIE, X) ./ η
ff_E_EFIE = potential(MWFarField3D(wavenumber=k), pts, j_EFIE, X)

@test norm(nf_E_EFIE - E.(pts))/norm(E.(pts)) ≈ 0 atol=0.01
@test norm(nf_H_EFIE - H.(pts))/norm(H.(pts)) ≈ 0 atol=0.01
@test norm(ff_E_EFIE - E.(pts, isfarfield=true))/norm(E.(pts, isfarfield=true)) ≈ 0 atol=0.01