# This file includes a function for defining the ordinary differential equation 
#       that describes the  Bouc-Wen model. \dot{f(t)} = boucwen(x, bwpar, t)

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    boucwen(x, bwpar)

    Bouc-Wen model
"""

@fastmath function boucwen(x, bwpar)

    # M, C, k0, df
    # INPUTS & OUTPUTS
    #---------------------------------------------------------------------
    # x:           system response at current time instant
    # bwpar: a structure of additional variales, vectors and matrices (including intermediate variables)
        # bwpar.M:      mass matrix
        # bwpar.C:      damping matrix
        # bwpar.k0:     lateral stiffness vector
        # bwpar.df:     right hand term of the equation of motion (expect for excitation)
        # bwpar.n_DoFs: number of degrees of freedom

        # parameters for Bouc-Wen model:
            # bwpar.alpha,  bwpar.A,        bwpar.n,    bwpar.beta,     bwpar.gamma
            # bwpar.d_nu,   bwpar.d_eta,    bwpar.p,    bwpar.q,        bwpar.d_psi
            # bwpar.lambda, bwpar.zeta_s,   bwpar.psi

        # intermediate variables:
            # bwpar.x1,      bwpar.x2,       bwpar.x3,     bwpar.x4,        bwpar.dx1
            # bwpar.dx2,     bwpar.nu,       bwpar.eta,    bwpar.zeta1,     bwpar.zeta2 
            # bwpar.zu,      bwpar.h,        bwpar.G0,     bwpar.G


    # number of degrees of freedom
    # n_DoFs = size(bwpar.M)[1]

    # displacement
    @views bwpar.x1 .= x[1:bwpar.n_DoFs]
    # velocity
    @views bwpar.x2 .= x[bwpar.n_DoFs+1:2*bwpar.n_DoFs]
    # hysteretic displacement
    @views bwpar.x3 .= x[2*bwpar.n_DoFs+1:3*bwpar.n_DoFs]
    # hysteretic energy
    @views bwpar.x4 .= x[3*bwpar.n_DoFs+1:end]

    # inter-story drift and velocity
    bwpar.dx1[1]      = bwpar.x1[1]
    @views bwpar.dx1[2:end] .= bwpar.x1[2:end] .- bwpar.x1[1:end-1]
    bwpar.dx2[1]      = bwpar.x2[1]
    @views bwpar.dx2[2:end] .= bwpar.x2[2:end] .- bwpar.x2[1:end-1]

    # inter-story nonlinear restoring force
    bwpar.nu         .= 1.0 .+ bwpar.d_nu .* bwpar.x4
    bwpar.eta        .= 1.0 .+ bwpar.d_eta .* bwpar.x4
    bwpar.zeta1      .= bwpar.zeta_s .* (1.0 .- exp.(-bwpar.p .* bwpar.x4))
    bwpar.zeta2      .= (bwpar.psi .+ bwpar.d_psi .* bwpar.x4) .* (bwpar.lambda .+ bwpar.zeta1)
    bwpar.zu         .= (bwpar.A ./ (bwpar.nu .* (bwpar.beta .+ bwpar.gamma))) .^ (1.0 ./ bwpar.n)

    # use loop to avoid extra allocation from exp.() and sign.()
    # bwpar.h          .= 1.0 .- bwpar.zeta1 .* exp.(-(bwpar.x3 .* sign.(bwpar.dx2) .-
    #                             bwpar.q .* bwpar.zu).^2 ./ (bwpar.zeta2 .^ 2))
    for k in 1:bwpar.n_DoFs
        bwpar.h[k]    = 1.0 - bwpar.zeta1[k] * exp(-(bwpar.x3[k] * sign(bwpar.dx2[k]) -
                                bwpar.q * bwpar.zu[k])^2 / (bwpar.zeta2[k] ^ 2))
    end

    bwpar.G0         .= bwpar.k0 .* (bwpar.alpha .* bwpar.dx1 .+ (1 .- bwpar.alpha) .* bwpar.x3)
    @views bwpar.G[1:end-1] .= bwpar.G0[1:end-1] .- bwpar.G0[2:end] 
    bwpar.G[end]      = bwpar.G0[end]

    # right hand of the equation
    bwpar.df[1:bwpar.n_DoFs]                  .= bwpar.x2
    #! excitation is considered outside this function
    # bwpar.df[n_DoFs+1:2*n_DoFs]   .= bwpar.M \ (F - bwpar.C*bwpar.x2 - bwpar.G) 
    bwpar.df[bwpar.n_DoFs+1:2*bwpar.n_DoFs]   .= bwpar.Minv * (-bwpar.C*bwpar.x2 .- bwpar.G) 

    bwpar.df[2*bwpar.n_DoFs+1:3*bwpar.n_DoFs] .= bwpar.h ./ bwpar.eta .* (bwpar.A .* bwpar.dx2 .- bwpar.nu .* 
                                                    (bwpar.beta .* abs.(bwpar.dx2) .* abs.(bwpar.x3) .^ (bwpar.n - 1) .* 
                                                    bwpar.x3 .+ bwpar.gamma .* bwpar.dx2 .* abs.(bwpar.x3) .^ bwpar.n))
    
    bwpar.df[3*bwpar.n_DoFs+1:end]            .= bwpar.x3 .* bwpar.dx2

    return bwpar.df
end