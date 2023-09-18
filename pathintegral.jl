# This file include a function for path integral and a sub-function for computing
#       the elements in the transition probability density matrix

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

"""
    pathintegral!(pdfmat, idcmat, idc_info, pis_par)
    
    Path integral for the PDF in next step
"""

@fastmath @inbounds function pathintegral!(pdfmat, idcmat, idc_info, pis_par)

    ## INPUTS:
    #-------------------------------------------------
    #  pdfmat:  probability density matrix at (h-1)-th time instant
    #  idcmat:  matrix of intrinsic drift coefficient
    #  idc_info: information related to intrinsic drift coefficient
    #  pis_par: some preallocated intermediate variables

    ## OUTPUTS:
    #-------------------------------------------------
    #  pdfmat:  probability density matrix at h-th time instant

    tpdstv = sqrt(idc_info.D * idc_info.dt)  # standard deviation of the normal distribution in TPD
    const1 = 1.0 / sqrt(2.0*pi) / tpdstv       # a constant for the pdf of normal distribution
    tpdvar2 = 1.0 / (2.0 * tpdstv^2)

    # extend the x-mesh for PCHIP interpolation
    pis_par.xlex[1]         = -pis_par.xl_fine[end] + pis_par.vl_fine[1] * idc_info.dt
    pis_par.xlex[end]       = -pis_par.xlex[1]
    pis_par.xlex[2:end-1]  .= pis_par.xl_fine

    # extend the function values for PCHIP interpolation
    pis_par.pchipfvals[1]   = 0.0
    pis_par.pchipfvals[end] = 0.0

    for j in eachindex(pis_par.vl_fine) # each row (fixed vj)
        # Interpolate the PDF(xi, vj, th-1) to PDF(xi-vj*dt, vj, th-1)
        @views pis_par.pchipfvals[2:end-1]  .= pdfmat[j, :]
        pchip_pdf = interpolate(pis_par.xlex, pis_par.pchipfvals, FritschButlandMonotonicInterpolation())
        @views pis_par.pdfmath[j, :] .= pchip_pdf.(pis_par.xmh[j, :])

        # Interpolate the IDC(xi, vj, th-1) to IDC(xi-vj*dt, vj, th-1)
        @views pis_par.pchipfvals[2:end-1]  .= idcmat[j, :]
        pchip_idc = interpolate(pis_par.xlex, pis_par.pchipfvals, LinearMonotonicInterpolation())
        @views pis_par.idcmath[j, :] .= pchip_idc.(pis_par.xmh[j, :])
    end

    # normalization of the interpolated PDF: PDF(xi-vj*dt, vj, th)
    pis_par.pdfmath .= max.(pis_par.pdfmath, 0.0) # PDF should be positive
    for i in 1:pis_par.n_vfine
        @views pis_par.pdfmath[i,:] .= pis_par.pdfmath[i,:] .* (sum(pdfmat[i,:]) ./ 
                                            (sum(pis_par.pdfmath[i,:]) + 1e-16))
    end

    # path integral solution for PDF(xi, vj, th)
    for i in 1:length(pis_par.xl_fine)  #for each column of PDF(xi, vj, th)
        # mean values of a row in the transition probability density (TPD) matrix
        @views pis_par.mean_tpd_xi .= pis_par.vl_fine .+ pis_par.idcmath[:, i] .* idc_info.dt

        # TPD matrix
        for j in 1:pis_par.n_vfine   # for a row of TPD matrix
            @inbounds for l in 1:pis_par.n_vfine
                pis_par.tpdmat[j, l] = tpdnormpdf(pis_par.vl_fine[j], pis_par.mean_tpd_xi[l], tpdvar2, const1)
            end
        end
        
        # normalize the TPD matrix
        pis_par.tpdmat .= max.(pis_par.tpdmat, 0.0) # PDF should be positive
        for k in eachindex(pis_par.vl_fine)
            @views tpd_vk = sum(pis_par.tpdmat[:, k]) * pis_par.dvl_fine
            if tpd_vk < 1e-16
                tpd_vk = 1.0
            end
            @views pis_par.tpdmat[:, k] .= pis_par.tpdmat[:, k] ./ tpd_vk
        end

        # calculate PDF(xi, vj, th)
        # 20230508 - replace matrix multiplication by dot multiplication
        pdfmat[:,i] .= 0.0
        for k in 1:pis_par.n_vfine
            @views pdfmat[:,i] .+= pis_par.tpdmat[:, k] .* pis_par.pdfmath[k, i] .*  pis_par.dvl_fine
        end
    end

    # normalize the PDF at the current time instant
    probck = sum(pdfmat) * pis_par.dxl_fine * pis_par.dvl_fine
    pdfmat .= pdfmat ./ max(probck, 1e-16)

end

@fastmath function tpdnormpdf(x::Float64, mu::Float64, tpdvar2::Float64, const1::Float64)

    ## a function to compute the probability density function values at points 
    ##      with given mean values and standard deviations

    # const1  = 1/sqrt(2.0*pi)/sigma
    # tpdvar2 = 1/2/sigma^2
    
    pdf = const1 * exp(-(x-mu)^2 * tpdvar2)

    return pdf
    
end