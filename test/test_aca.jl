using Test
using FastBEAST
using LinearAlgebra
using StaticArrays
using BenchmarkTools

N = 1000
A = rand(N, N)

##
@views function fct(B, x, y)
    #for i in eachindex(x)
    #    for j in eachindex(y)
            #B[i, j] = A[x[i],y[j]]
            B[:,:] = A[x, y]
    #    end
    #end
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

##
function mytest(lm, b)
    b[1:200] = lm[1:200, 1]
    return
end

function mytest2(lm, b, I, J)
    lm.μ(b, lm.τ[I], lm.σ[J])
    return
end

##
b = zeros(Float64, 200)
@btime mytest($lm, $b)
@show b

B = zeros(Float64, 200, 1)
@btime mytest2($lm, $B, 1:200, 1:1)
@show B


##

N = 1000
A = rand(N,N)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

A = U*diagm(S)*V'

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)

@test U*V ≈ A atol = 1e-14

##

Ns = 200
Nt = 100
A = rand(Nt,Ns)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)

@test U*V ≈ A atol = 1e-14

##

Ns = 100
Nt = 200
A = rand(Nt,Ns)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)

@test U*V ≈ A atol = 1e-14

##

a = [1.0 0.0 0.0 0.0 0.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)

@test U*V == A

##
a = [1.0 -2.0 6.0 4.0 5.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)

@test U*V == A

##
a1 = [1.0 -2.0 6.0 4.0 5.0]
b1 = [7.0 3.0 1.0 9.0 1.0]
a2 = [9.0 3.0 -6.0 2.0 3.0]
b2 = [5.0 4.0 3.0 1.0 -4.0]
#a3 = [10.0 2.0 -6.0 2.0 3.0]
#b3 = [-11.0 4.0 3.0 1.0 -4.0]
A = a1'*b1 + a2'*b2 #+ a3'*b3

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)

@test U*V ≈ A atol = 1e-13

##
Ns = 100
Nt = 200
A = rand(Nt,Ns)

B = zeros(2*Nt, 2*Ns)

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

rowindices = [2*Nt - 2*i + 1 for i=1:Nt]
colindices = [2*Ns - 2*i + 1 for i=1:Ns]

for i = 1:Ns
    for j=1:Nt
        B[end-2*j+1,end-2*i+1] = A[j,i]
    end
end

function fct(C, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            C[i,j] = B[x[i],y[j]]
        end
    end
end

lm = LazyMatrix(fct, rowindices, colindices, Float64)

U, V = aca(lm)

@test U*V ≈ A atol = 1e-14

##
A = rand(2,1)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)
@test U*V ≈ A atol = 1e-14

##
A = rand(1,1)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm)
@test U*V ≈ A atol = 1e-14

##
N = 100
A = zeros(3*N,3*N)

val = rand(N,N)
val2 = rand(N,N)

for i = 1:N
    for j = 1:N
        A[i,j]=val[i,j]
        A[i+N,j+N]=val2[i,j]
        A[i+2*N,j+2*N]=val[i,j]
    end
end

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:3*N ]

A = U*diagm(S)*V'

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm, isblockstructured=true)

@test U*V ≈ A atol = 1e-14