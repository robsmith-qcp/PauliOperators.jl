using LinearAlgebra
using KrylovKit
using VectorInterface

function VectorInterface.inner(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}) where N
    bra = pv1.ps
    ket = pv2.ps
    new_ps = adjoint(bra)*ket
    return tr(new_ps)
end

function KrylovKit.norm(pv::VectorizedPauliSum{N}) where N
    pv1 = deepcopy(pv)
    pv2 = deepcopy(pv)
    return real.(LinearAlgebra.dot(pv1, pv2))
end

function LinearAlgebra.dot(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}) where N
    bra = pv1.ps
    ket = pv2.ps
    new_ps = adjoint(bra)*ket
    return tr(new_ps)
end

function VectorInterface.scale(pv::VectorizedPauliSum{N}, a::Float64) where N
    return VectorizedPauliSum(a*pv.ps)
end

function VectorInterface.scale(pv::VectorizedPauliSum{N}, a::ComplexF64) where N
    return VectorizedPauliSum(a*pv.ps)
end

function VectorInterface.add!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{M}, a::Float64) where {N,M}
    tmp_ax = a*pv2.ps
    tmp_r = pv1.ps + tmp_ax
    return VectorizedPauliSum(tmp_r)
end 

function VectorInterface.add!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{M}, a::ComplexF64) where {N,M}
    tmp_ax = a*pv2.ps
    tmp_r = pv1.ps + tmp_ax
    return VectorizedPauliSum(tmp_r)
end

function VectorInterface.zerovector(pv::VectorizedPauliSum{N}) where N
    tmp_zv = 0.0*pv.ps
    return VectorizedPauliSum(tmp_zv)
end

function VectorInterface.scale!!(pv::VectorizedPauliSum{N}, a::Float64) where N
    tmp_ps = a*pv.ps
    return VectorizedPauliSum(tmp_ps)
end

function VectorInterface.scale!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Int) where N
    linear_combo = a*pv1.ps + a*pv2.ps
    return VectorizedPauliSum(linear_combo)
end

function VectorInterface.add!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Int64, b::Float64) where N
    tmp_ax = a*pv1.ps
    tmp_by = b*pv2.ps
    tmp_r = tmp_ax + tmp_by
    return VectorizedPauliSum(tmp_r)
end

function VectorInterface.add!!(pv1::VectorizedPauliSum{N}, pv2::VectorizedPauliSum{N}, a::Int64, b::ComplexF64) where N
    tmp_ax = a*pv1.ps
    tmp_by = b*pv2.ps
    tmp_r = tmp_ax + tmp_by
    return VectorizedPauliSum(tmp_r)
end
