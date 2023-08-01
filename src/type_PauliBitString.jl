"""
In this representation, the Pauli string operator is represented as two binary strings, one for x and one for z.

The format is as follows: Z^z1 X^x1 ⊗ Z^z2 X^x2 ⊗ ⋯ ⊗ Z^zN X^xN  
    
Products of operators simply concatonate the left and right strings separately. For example, 

    XYZIY = 11001|01101
"""
struct PauliBitString{N} <: Integer
    θ::UInt8
    z::Int128
    x::Int128
end

"""
    PauliBitString(z::I, x::I) where I<:Integer

TBW
"""
function PauliBitString(z::I, x::I) where I<:Integer
    N = maximum(map(i -> ndigits(i, base=2), [x, z]))
    θ = count_ones(z & x)*3 % 4
    return PauliBitString{N}(θ, z, x)
end

"""
    PauliBitString(str::String)

TBW
"""
function PauliBitString(str::String)
    for i in str
        i in ['I', 'Z', 'X', 'Y'] || error("Bad string: ", str)
    end

    x = Int128(0)
    z = Int128(0)
    ny = 0 
    N = length(str)
    idx = Int128(1)
    two = Int128(2)
    one = Int128(1)

    for i in str
        # println(i, " ", idx, typeof(idx))
        if i in ['X', 'Y']
            x |= two^(idx-one)
            if i == 'Y'
                ny += 1
            end
        end
        if i in ['Z', 'Y']
            z |= two^(idx-one)
        end
        idx += 1
    end
    θ = 3*ny%4
    return PauliBitString{N}(θ, z,x) 
end


"""
    PauliBitString(N::Integer; X=[], Y=[], Z=[])

constructor for creating PauliBoolVec by specifying the qubits where each X, Y, and Z gates exist 
"""
function PauliBitString(N::Integer; X=[], Y=[], Z=[])
    for i in X
        i ∉ Y || throw(DimensionMismatch)
        i ∉ Z || throw(DimensionMismatch)
    end
    for i in Y
        i ∉ Z || throw(DimensionMismatch)
    end
    
    str = ["I" for i in 1:N]
    for i in X
        str[i] = "X"
    end
    for i in Y
        str[i] = "Y"
    end
    for i in Z
        str[i] = "Z"
    end
   
    # print(str[1:N])
    return PauliBitString(join(str))
    
end
