# This function include functions for generating mesh grids for one-dimensional 
#       or multi-dimensional box domain.

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    gridknots(lb::VecOrMat, ub::VecOrMat, n::VecOrMat{Int64})

    Generate multi-dimensional mesh grids

    gridknots(lb::Real, ub::Real, n::Int)

    Generate one-dimensional mesh grids
"""

function gridknots(lb::VecOrMat, ub::VecOrMat, n::VecOrMat{Int64})
    
    n_dim   = length(lb)

    n_knots = prod(n)

    knots   = zeros(Float64, n_dim, n_knots)

    nn1 = 1

    for i in 1:n_dim
        ni  = n[i]

        nn0 = nn1

        nn1 = nn1 * ni

        nn2 = div(n_knots,nn1)

        dxi = (ub[i] - lb[i]) / (ni - 1)
        xiv = collect(lb[i]:dxi:ub[i])

        for j in 1:nn2
            j1 = (j-1) * nn1 + 1
            j2 = j * nn1
            knots[i,j1:j2] .= kron(xiv, ones(Float64,nn0))
        end
        
    end

    return knots

end


function gridknots(lb::Real, ub::Real, n::Int)

    knots    = zeros(Float64, 1, n)
    dxi      = (ub - lb) / (n - 1)
    knots[1,:]   .= collect(lb:dxi:ub)

    return knots

end
