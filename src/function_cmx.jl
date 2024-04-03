using PauliOperators
using LinearAlgebra

function cmx(order::Int,A::PauliSum{N},Ψ::Vector{ComplexF64}) where N
        μ = zeros(ComplexF64, order)
        expect = Dict()
        op = Dict()
        op[1] = A
        expect[1] = (Ψ'*(op[1]*Ψ))[1]
        if (order > 1)
                for i in 2:order
                        op[i] = op[i-1]*A
                        expect[i] = (Ψ'*(op[i]*Ψ))[1]
                end
        end
        for j in 1:order
                μ[j] = expect[j]
                if (j > 1)
                        for k in 0:(j-2)
                                μ[j] -= binomial(j-1, k) * μ[k+1] * expect[j-k-1]
                        end
                end
        end
        return μ
end
