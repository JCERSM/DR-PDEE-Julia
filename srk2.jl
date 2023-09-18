# This file defines a function for 2-order stochastic Runge-Kutta scheme

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    srk2!(df, g, par, ini_val, D, dt, nt, sol)

    Solve the equation of motion under stochastic excitations by 2-order stochastic Runge-Kutta scheme
"""

@fastmath function srk2!(df, g, par, ini_val, D, dt, nt, sol)

    # dX = df(X)*dt + g*dW

    # df:   right hand term of the equation
    # g:    right hand term of the equation
    # par:  a structure of additional parameters for equation of motion
    # ini_val: initial value of the state vector
    # dt:   time step of the excitation
    # D*dt: intensity of the Gaussian white noise excitation
    # nt:   number of solution
    # sol:  initialized solution (A nt+1 column matrix. Each column is the solution at a time instant.)
    #        This is also the solution to the equation

    # initial values
    sol[:, 1] .= ini_val
    # fi         = deepcopy(ini_val)

    # number of DoFs
    n_DoFs     = length(ini_val)

    # intensity of the Gaussian white noise
    amp        = sqrt(D*dt);

    # temporary variables
    dW_g       = zeros(Float64, n_DoFs)
    k1         = zeros(Float64, n_DoFs)
    k2         = zeros(Float64, n_DoFs)

    # for each time step
    @inbounds for i in range(1,nt-1)
        dW_g      .= g .* amp .* randn(Float64)    #excitation

        @views k1 .= df(sol[:, i], par)
        @views k2 .= df(sol[:, i] .+ k1.*dt .+ dW_g, par)

        @views sol[:, i+1] .= sol[:, i] .+ (k1 .+ k2) ./ 2.0 .* dt .+ dW_g

        # sol[:, i+1] .= fi
    end

end