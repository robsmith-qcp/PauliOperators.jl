using PauliOperators
using LinearAlgebra
#using Plots
#using Plots.PlotMeasures
using BenchmarkTools
using OrderedCollections

function operator_clustering(operator, clusters)
    op_cl = OrderedDict{Vector{FixedPhasePauli},ComplexF64}()
    for (idx,(op,coeff)) in enumerate(operator.ops)
        tmp_str = uppercase(string(op))
        paulis = Vector{FixedPhasePauli}()
        for cluster in clusters
            new_pauli = FixedPhasePauli(tmp_str[cluster])
            push!(paulis,new_pauli)
        end
        op_cl[paulis] = coeff
    end
    return op_cl
end

function cmx_c(order,A,Ψ,clusters)
    μ = zeros(ComplexF64, 2*order+1)
    expect = Dict{Int,ComplexF64}()
    op = Dict{Int,PauliSum}()
	  clip!(A)
	  As = operator_clustering(A, clusters)
    op[1] = A
	  ev1 = 0.0
	  for (key,val) in As
		    for (paulis,coeff) in val
			      evp = coeff 
			      for i in 1:length(clusters)
				        evp *= (Ψ[i]'*(paulis[i]*Ψ[i]))[1]
			      end
			      ev1 += evp
		    end
	  end
    expect[1] = ev1
    for j in 2:2*order+1
        op[j] = op[j-1]*A
        clip!(op[j])
        println("Operator order ", j, " with size ", length(op[j]))
		    ops = operator_clustering(op[j], clusters)
		    evj = 0.0
		    println("Computing expectation values for operator order ", j)
		    for (paulis,coeff) in val
            evp = coeff
            for k in 1:length(clusters)
                evp *= (Ψ[k]'*(paulis[k]*Ψ[k]))[1]
            end
            evj += evp
			  end
        expect[j] = evj
    end
    μ[1] = expect[1]
    println("CMF = ", μ[1])
	  for j in 2:2*order+1
        μ[j] = expect[j]
        for k in 0:j-2
            μ[j] -= binomial(j-1, k) * μ[k+1] * expect[j-k-1]
        end
    end
    E₀ = expect[1]
    v = μ[2:order+1]
    M = zeros(ComplexF64, order, order)
    for row in 1:order
        for col in 1:order
            M[row, col] = μ[row + col + 1]
        end
    end
	  Minv = inv(M)
	  E₀ -= (v'*(Minv*v))[1]
    return μ, E₀, expect[1]
end
