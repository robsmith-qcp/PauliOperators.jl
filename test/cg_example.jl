using PauliOperators
using BlockDavidson
using LinearAlgebra
using KrylovKit
using VectorInterface
using Printf


# Initial test with complex vectors and matrix

alg = CG(; maxiter=500, tol=1e-9, verbosity=3)

x₀ = rand(Complex{Float64}, 10)
b = rand(Complex{Float64}, 10)
A = rand(Complex{Float64}, 10, 10)
A = A + adjoint(A)

x, info = linsolve(A, b, x₀, alg)
display(x)


# Testing with vectorized Pauli operators and a superoperator in Liouville space

nq = 3
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
H₀, V = build_ops(nq)
sw = 1.0im*V

#println("Unperturbed Hamiltionian")
#display(H₀)
#println("Perturbation")
#display(V)
#println("Guess anti-Hermitian")
#display(sw)

b = VectorizedPauliSum(-1.0*V)
L = PauliOperators.vectorized_commutator(H₀)
#wrapvec(b) = b
wrapop(::VectorizedPauliSum) = a.ps
x₀ = VectorizedPauliSum(sw)

function testingmatvec(x)
    return L*x
end

alg = CG(; maxiter=500, tol=1e-9, verbosity=3)
x, info = linsolve(testingmatvec, b, x₀, alg)

display(x)

