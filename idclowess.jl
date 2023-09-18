# This file includes a function for Locally Weighted Scatterplot Smoothing (LOWESS)
#       and several sub-functions.

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    idclowess(pts, vals, knots, lowess_par, n_locpts)

    Evaluate the intrinsic drift coefficient by KNN and LOWESS at all mesh knots
"""

@fastmath @inbounds function idclowess(pts, vals, knots, lowess_par, n_locpts)

    ## INPUTS
    #-----------------------------------------------------------------------
    #  pts:  observation points
    #  vals: function values at pts
    #  knots: mesh grids at which intrinsic drift coefficient to be computed
    #  lowess_par: data structure for idc calculation by LOWESS (2D)
    #  n_locpts: number of points to be used in LOWESS
    
    ## LOWESS: locally weighted smoothing scatterplot
    #-----------------------------------------------------------------------
    ## Reference:
    ## Cleveland WS (1979) Robust locally weighted regression and smoothing scatterplots.
    ##    Journal of the American Statistical Association 74: 829-836.

    # standard deviation of the points
    lowess2d_stdv!(pts, lowess_par.stdvs)

    # normalize the points
    lowess_par.pts_n   .= pts ./ lowess_par.stdvs
    lowess_par.knots_n .= knots ./ lowess_par.stdvs

    # bulid a kd-tree
    tree            = KDTree(lowess_par.pts_n)

    # KDTree
    @views idxs_ns, dists_ns  = knn(tree, lowess_par.knots_n, n_locpts, true)

    # evaluate the intrinsic drift coefficient at each mesh grid
    for k in 1:lowess_par.n_knots
        @views lowess_par.locpts       .= pts[:, idxs_ns[k]]
        @views dist0_n                  = dists_ns[k][end] + 1e-12
        @views lowess_par.locvals      .= vals[idxs_ns[k]]
        lowess2d_weight!(dists_ns[k], dist0_n, n_locpts, lowess_par.weights)
        @views lowess2d_linearbasis!(knots[:,k], lowess_par.lb_tmp)
        lowess_par.idc[k]  = sum((lowess_par.lb_tmp .* localregression(lowess_par.weights, lowess_par.locpts, 
                                        lowess_par.locvals, n_locpts, lowess_par.A, lowess_par.b, 
                                        lowess_par.re_coeff)))
    end

    return lowess_par.idc

end


# a sub-function to compute weight of points
@fastmath function lowess2d_weight!(dists::Vector{Float64}, dist0::Float64, 
                          n_locpts::Int64, weights::Vector{Float64})
    @inbounds for i in 1:n_locpts
        weights[i] = max((1.0 - (dists[i] / dist0) ^ 3) ^ 3, 0.0)
    end
end

# a sub-function to calculate the standard deviation
@fastmath function lowess2d_stdv!(pts::AbstractArray{Float64}, stdv::MVector{2,Float64})
    stdv .= 0
    @views stdv[1] = max(std(pts[1,:]), 1e-10)
    @views stdv[2] = max(std(pts[2,:]), 1e-10)
end

# a sub-function to generate linear bases for regression
@fastmath function lowess2d_linearbasis!(pts::AbstractArray, 
                               lin_bas_func_val::MVector{3,Float64})
    lin_bas_func_val .= 0.0
    lin_bas_func_val[1] = 1.0
    lin_bas_func_val[2] = pts[1]
    lin_bas_func_val[3] = pts[2]
end

# a sub-function to perform regression
@fastmath function localregression(w::Vector{Float64}, locpts::Matrix{Float64}, f::Vector{Float64}, 
                         n_locpts::Int64, A::MMatrix{3,3,Float64}, b::MVector{3,Float64}, re_coeff::MVector{3,Float64})

    # coefficient matrix
    # A must be a 3*3 zero matrix
    A .= 0.0
    b .= 0.0
    # re_coeff .= 0.0

    @inbounds for i in 1:n_locpts
        A[1, 1] += w[i]
        A[2, 1] += w[i] * locpts[1, i]
        A[3, 1] += w[i] * locpts[2, i]
        A[2, 2] += w[i] * locpts[1, i]^2
        A[3, 2] += w[i] * locpts[1, i] * locpts[2, i]
        A[3, 3] += w[i] * locpts[2, i]^2
    end
    # A[1, 2] = A[2, 1]
    # A[1, 3] = A[3, 1]
    # A[2, 3] = A[3, 2]

    # coefficient vector
    @inbounds for i in 1:n_locpts
        b[1] += w[i] * f[i]
        b[2] += w[i] * f[i] * locpts[1, i]
        b[3] += w[i] * f[i] * locpts[2, i]
    end
    
    # solve the equation to obtain regression coefficient
    # re_coeff = A \ b

    c1          =    A[1,1] * (A[2,2]*A[3,3] - A[3,2]*A[3,2]) -
                     A[2,1] * (A[2,1]*A[3,3] - A[3,1]*A[3,2]) +
                     A[3,1] * (A[2,1]*A[3,2] - A[3,1]*A[2,2])

    c1          = max(c1, 1e-10)
    
    re_coeff[1] =  ((A[2,2]*A[3,3] - A[3,2]*A[3,2]) * b[1] +
                    (A[3,1]*A[3,2] - A[2,1]*A[3,3]) * b[2] +
                    (A[2,1]*A[3,2] - A[3,1]*A[2,2]) * b[3]) / c1

    re_coeff[2] =  ((A[3,2]*A[3,1] - A[2,1]*A[3,3]) * b[1] +
                    (A[1,1]*A[3,3] - A[3,1]*A[3,1]) * b[2] +
                    (A[2,1]*A[3,1] - A[1,1]*A[3,2]) * b[3]) / c1
    
    re_coeff[3] =  ((A[2,1]*A[3,2] - A[2,2]*A[3,1]) * b[1] +
                    (A[2,1]*A[3,1] - A[1,1]*A[3,2]) * b[2] +
                    (A[1,1]*A[2,2] - A[2,1]*A[2,1]) * b[3]) / c1

    return re_coeff
end
