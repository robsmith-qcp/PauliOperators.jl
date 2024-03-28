using LinearAlgebra
using KrylovKit
using VectorInterface

function LinearAlgebra.mul!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Number) where N
    scaledps = a*pv2.ps
    new_pauli = scaledps
    return VectorizedPauliSum(new_pauli)
end

function LinearAlgebra.rmul!(pv::VectorizedPauliSum{N}, a::Number) where N
    new_ps = a*pv.ps
    return VectorizedPauliSum(new_ps)
end

function LinearAlgebra.axpy!(a::Number, pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}) where N
    scaledps = a*pv1.ps
    new_ps = scaledps + pv2.ps
    return VectorizedPauliSum(new_ps)
end

function LinearAlgebra.axpby!(a::Number, pv1::VectorizedPauliSum{N}, b::Number, pv2::VectorizedPauliSum{N}) where N
    new_ps = a*pv1.ps + b*pv2.ps
    return VectorizedPauliSum(new_ps)
end

function LinearAlgebra.dot(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}) where N
    bra = pv1.ps
    ket = pv2.ps
    new_ps = adjoint(bra)*ket
    return tr(new_ps)
end

function LinearAlgebra.norm(pv::VectorizedPauliSum{N}) where N
    pv1 = deepcopy(pv)
    pv2 = deepcopy(pv)
    inner_prod = LinearAlgebra.dot(pv1, pv2)
    return sqrt(inner_prod)
end

function Base.isless(a::ComplexF64, b::ComplexF64)
    a_new = convert(Float64, a*adjoint(a))
    b_new = convert(Float64, b*adjoint(b))
    return a_new < b_new
end

function VectorInterface.inner(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}) where N
    bra = pv1.ps
    ket = pv2.ps
    new_ps = adjoint(bra)*ket
    return tr(new_ps)
end

function VectorInterface.add!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Number) where N
    tmp_ax = a*pv2.ps
    new_ps = pv1.ps + a*pv2.ps
    return VectorizedPauliSum(new_ps)
end 

function VectorInterface.add!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Number, b::Number) where N
    tmp_ax = a*pv1.ps
    tmp_by = b*pv2.ps
    tmp_r = tmp_ax + tmp_by
    return VectorizedPauliSum(tmp_r)
end

function VectorInterface.zerovector(pv::VectorizedPauliSum{N}) where N
    tmp_zv = 0.0*pv.ps
    return VectorizedPauliSum(tmp_zv)
end

function VectorInterface.scale(pv::VectorizedPauliSum{N}, a::Number) where N
    return VectorizedPauliSum(a*pv.ps)
end

function VectorInterface.scale!!(pv::VectorizedPauliSum{N}, a::Float64) where N
    tmp_ps = a*pv.ps
    return VectorizedPauliSum(tmp_ps)
end

function VectorInterface.scale!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Int) where N
    linear_combo = a*pv1.ps + a*pv2.ps
    return VectorizedPauliSum(linear_combo)
end

