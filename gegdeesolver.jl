# This file includes a function for solving the Globally-Evolving-based 
#       Generalized Probability Density Evolution Equation (GE-GDEE) or the Dimension-Reduced
#       Probability Density Evolution Equation (DR-PDEE) by using path integral at each time step

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    gegdeesolver!(idc_ts, idc_info, pdfmat_ini, pis_par, gegdee_temp)

    Solve the DR-PDEE/GE-GDEE by path integral
"""

@fastmath @inbounds function gegdeesolver!(idc_ts, idc_info, pdfmat_ini, pis_par, gegdee_temp)

    #### INPUTS:
    ## idc_ts: matrix of intrinsic drift coefficients (each column at a time instant)
    ## idc_info: a structure including necessary information about idc_ts
    ## pdfmat_ini: initial value of the GE-GDEE
    ## pis_par: a structure for path integral
    ## gegdee_temp: a structure including intermediate variable for solving GE-GDEE

    flagshow = 2000

    # reset initial value of pdfmat
    gegdee_temp.pdfmat .= pdfmat_ini

    # probability density function at the initial time instant
    #gegdee_temp.pdf_x[:, 1] .= sum(pdfmat_ini[:,i], dims=1)' .* pis_par.dvl_fine
    for i in 1:pis_par.n_xfine
        @views gegdee_temp.pdf_x[i, 1] = sum(pdfmat_ini[:, i]) .* pis_par.dvl_fine
    end
    # gegdee_temp.pdf_v[:, 1] .= sum(pdfmat_ini, dims=2) .* pis_par.dxl_fine
    for i in 1:pis_par.n_vfine
        @views gegdee_temp.pdf_v[i, 1] = sum(pdfmat_ini[i, :]) .* pis_par.dxl_fine
    end

    # perform path integral at each time instants
    for k in 1:idc_info.n_step-1
        # reshape the intrinsic drift coefficient in to matrix: n_xcrude * n_vcrude
        @views idc_crude    = reshape(idc_ts[:, k], idc_info.n_xcrude, idc_info.n_vcrude)
        if idc_info.n_xcrude == pis_par.n_xfine && idc_info.n_vcrude == pis_par.n_vfine
            transpose!(gegdee_temp.idcmat, idc_crude)
        else
            # interpolation is performed only when the mesh of IDC is differnt from the mesh of path integral
            # interpolate the intrinsic drift coefficient from crude mesh to fine mesh
            interp_linear       = linear_interpolation((idc_info.xl_crude, idc_info.vl_crude), idc_crude)
            gegdee_temp.idcmat .= interp_linear.(pis_par.xm_fine, pis_par.vm_fine)
        end

        # path integral
        pathintegral!(gegdee_temp.pdfmat, gegdee_temp.idcmat, idc_info, pis_par)
    
        # probability density function at current time instant
        for i in 1:pis_par.n_xfine
            @views gegdee_temp.pdf_x[i, k+1] = sum(gegdee_temp.pdfmat[:, i]) .* pis_par.dvl_fine
        end
        for i in 1:pis_par.n_vfine
            @views gegdee_temp.pdf_v[i, k+1] = sum(gegdee_temp.pdfmat[i, :]) .* pis_par.dxl_fine
        end

        if mod(k,flagshow) == 0
            @printf("Evaluating the PDF at the %d-th/%d time step.\n", k, idc_info.n_step-1)
        end
    end

    return gegdee_temp.pdf_x, gegdee_temp.pdf_v
end