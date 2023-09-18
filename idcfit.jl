# This file includes a function for determining the intrinsic drift coefficient (IDC)
#       at each time step by Locally Weighted Scatterplot Smoothing (LOWESS)

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    idcfit!(idc_input, n_locpts, lowess_par, idc_temp)

    fit the intrinsic drift coefficients using a single thread
"""

@fastmath @inbounds function idcfit!(idc_input, n_locpts, lowess_par, idc_temp)

    ### INPUTS:
    ##------------------------------------------------------------------------
    # idc_input: structured data including responses and konts of interest (crude mesh)
    # n_locpts: number of locally selected point for fitting the intrinsic drift coefficients
    # lowess_par: structured data for fitting the intrinsic drift coefficient by LOWESS
    # idc_temp: structured data defining intermediate variables for fitting idcs

    flagshow = 2000

    # number of samples
    N = idc_input.n_samp


    # double the data according to its symmetry
    n_locpts = n_locpts * 2
    for k in 1:idc_input.nt-1
        @views idc_temp.pts[1, 1:N]     .=  idc_input.disp_samp[:, k+1]
        @views idc_temp.pts[1, N+1:2*N] .= -idc_input.disp_samp[:, k+1]
        @views idc_temp.pts[2, 1:N]     .=  idc_input.velo_samp[:, k+1]
        @views idc_temp.pts[2, N+1:2*N] .= -idc_input.velo_samp[:, k+1]
        @views idc_temp.fs[1:N]         .=  idc_input.nforce_samp[:, k+1]
        @views idc_temp.fs[N+1:2*N]     .= -idc_input.nforce_samp[:, k+1]
    
        idc_temp.idc_ts[:, k] .= idclowess(idc_temp.pts, idc_temp.fs, idc_input.knots, lowess_par, n_locpts)
    
        if mod(k,flagshow) == 0
            @printf("Evaluating the IDC at the %d-th/%d time step.\n", k, idc_input.nt-1)
        end
    end

    return idc_temp.idc_ts

end