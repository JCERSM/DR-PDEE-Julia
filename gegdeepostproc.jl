# This file includes functions for post-process of the solution of the GE-GDEE/DR-PDEE:
#       (1) Cumulative distribution function
#       (2) Mean value process
#       (3) Variance process
#       (4) Skewness process
#       (5) Kurtosis process

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn


# evaluate the cumulative distribution function
function cdfgegdee(pdf_x, pdf_v, dxl, dvl)
    
    # CDF of displacement response
    cdf_x = cumsum(pdf_x, dims=1) * dxl
    # CDF of velocity response
    cdf_v = cumsum(pdf_v, dims=1) * dvl
    
    return cdf_x, cdf_v

end

# evluate the mean value process
function meangegdee(pdf_x, pdf_v, xl, vl, dxl, dvl)
    
    # mean value process of displacement response
    mean_x = sum(xl .* pdf_x, dims = 1) .* dxl
    # mean value process of velocity response
    mean_v = sum(vl .* pdf_v, dims = 1) .* dvl

    return mean_x, mean_v

end

# evaluate the variance process
function vargegdee(pdf_x, pdf_v, mean_x, mean_v,  xl, vl, dxl, dvl)
    
    # variance process of displacement response
    var_x = sum((xl .- mean_x).^2 .* pdf_x, dims = 1) .* dxl
    # variance process of velocity response
    var_v = sum((vl .- mean_v).^2 .* pdf_v, dims = 1) .* dvl

    return var_x, var_v
    
end

# evaluate the skewness process
function skewgegdee(pdf_x, pdf_v, mean_x, mean_v, stdv_x, stdv_v, xl, vl, dxl, dvl)
    
    # skewness process of displacement response
    skew_x = sum(((xl .- mean_x) ./ stdv_x) .^3 .* pdf_x, dims = 1) .* dxl
    # skewness process of velocity response
    skew_v = sum(((vl .- mean_v) ./ stdv_v) .^3 .* pdf_v, dims = 1) .* dvl

    return skew_x, skew_v
    
end


# evaluate the kurtosis process
function kurtgegdee(pdf_x, pdf_v, mean_x, mean_v, stdv_x, stdv_v, xl, vl, dxl, dvl)
    
    # kurtosis process of displacement response
    kurt_x = sum(((xl .- mean_x) ./ stdv_x) .^4 .* pdf_x, dims = 1) .* dxl
    # kurtosis process of velocity response
    kurt_v = sum(((vl .- mean_v) ./ stdv_v) .^4 .* pdf_v, dims = 1) .* dvl

    return kurt_x, kurt_v
    
end


# emprical cummulative distribution function
function mcsecdfs(resamples, resmesh)
    
    (np, nt) = size(resamples)

    nx = length(resmesh)

    ecdfmat = zeros(Float64, nx, nt)

    for i in 1:nt
        resti = resamples[:, i]
        for j in 1:nx
            ecdfmat[j,i] = Float64(count(<=(resmesh[j]), resti)) ./ Float64(np)
        end
    end

    return ecdfmat
end