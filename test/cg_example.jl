using PauliOperators
using BlockDavidson
using LinearAlgebra
using KrylovKit
using VectorInterface
using Printf


# Initial test with complex vectors and matrix

#alg = CG(; maxiter=500, tol=1e-9, verbosity=3)

#x₀ = rand(Complex{Float64}, 10)
#b = rand(Complex{Float64}, 10)
#A = rand(Complex{Float64}, 10, 10)
#A = A + adjoint(A)

#x, info = linsolve(A, b, x₀, alg)
#display(x)


# Testing with vectorized Pauli operators and a superoperator in Liouville space

function build_ops(nq)
        H₀ = random_Pauli(nq)
        for i in 1:64
                H₀ += 0.01*random_Pauli(nq)
        end
        H₀ = H₀ + adjoint(H₀) 

        V = random_Pauli(nq)
        for i in 1:64
                V += 0.01*random_Pauli(nq)
        end
        V = V + adjoint(V)
        clip!(H₀)
        clip!(V)
        return H₀, V
end

function vectorize_superop(H::PauliSum{N}; T=ComplexF64) where N
    function mymatvec(v::VectorizedPauliSum{N})
        return VectorizedPauliSum(liouville_space_commutator(H, v.ps))
    end
    return PauliOperators.SuperOperator{T, N}(mymatvec, 2^N, false)
end

function liouville_space_commutator(ps1::PauliSum{N}, ps2::PauliSum{N}) where N
    out = ps1*ps2 - transpose(ps1)*ps2
    return out
end

function testingmatvec(x)
    return L*x
end

nq = 3
H₀, V = build_ops(nq)
S = 1.0im*V

#println("Unperturbed Hamiltionian")
#display(H₀)
#println("Perturbation")
#display(V)
#println("Guess anti-Hermitian")
#display(sw)

b = VectorizedPauliSum(-1.0*V)
L = vectorize_superop(H₀)
x₀ = VectorizedPauliSum(S)

alg = CG(; maxiter=500, tol=1e-9, verbosity=3)
x, info = linsolve(testingmatvec, b, x₀, alg)

display(x)

