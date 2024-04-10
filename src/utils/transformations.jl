
# The transformation matrix from complex SHs to real SHs
function ctran(L)
    AA = zeros(ComplexF64, 2L+1, 2L+1)
    for i = 1:2L+1
        for j in [i, 2L+2-i]
            # @show i,j
            AA[i,j] = begin
                if i == j == L+1
                    1
                elseif i > L+1 && j > L+1
                    (-1)^(i-L-1)/sqrt(2)
                elseif i < L+1 && j < L+1
                    im/sqrt(2)
                elseif i < L+1 && j > L+1
                    (-1)^(i-L)/sqrt(2)*im
                elseif i > L+1 && j < L+1
                    1/sqrt(2)
                end
            end
            # @show AA[i,j]
        end
    end
    return AA
end