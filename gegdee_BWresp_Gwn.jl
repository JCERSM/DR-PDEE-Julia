# A manuscript for evaluating the stochastic response of Bouc-Wen system under Gaussian white noises

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

##########################################################################################
## Load necessary packages
##########################################################################################
using Distributions
using Interpolations
using LinearAlgebra
using NearestNeighbors
using Plots
using Printf
using Random
using Statistics
using StaticArrays
include("./DRPDEE.jl")
using .DRPDEE

##########################################################################################
## Structural model
##########################################################################################
# number of degrees of freedom
n_DoFs     = 10
# DoF of interest
iDoF       = 10
# number of columns
str_colnum = 3
# Lumped masses (kg) [dof by 1 vector]
str_m      = ones(Float64, n_DoFs) * 2.6e5;
# Height of each story (m) [dof by 1 vector]
str_h      = [4.0; 3.0; 3.0; 3.0; 3.0; 3.0; 3.0; 3.0; 3.0; 3.0]
# Length of column section of each story (m) [dof by 1 vector]  
str_a      = [0.5; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4]
# Width of column section of each story (m) [dof by 1 vector]
str_b      = [0.5; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4; 0.4]
# Initial Young's modulus of each story (Pa) [dof by 1 vector]
str_E      = ones(Float64, n_DoFs) * 3e10;
# Damping ratios of the first two modes [2 by 1 vector]
zeta       = [0.05; 0.05]

# Moment of inertia of column section of each story (m4) [dof by 1 vector]
str_I      = str_b .* str_a .^ 3.0 / 12.0
# Initial lateral inter-story stiffness of each story (N/m) [dof by 1 vector]
str_k      = str_colnum .* 12.0 .* str_E .* str_I ./ str_h.^3
# Lumped mass (kg) [dof by dof matrix]
M          = diagm(str_m)
Minv       = diagm(1 ./ str_m)
# Initial lateral inter-story stiffness (N/m) [dof by dof matrix]
K          = SymTridiagonal(str_k + [str_k[2:end]; 0], -str_k[2:end])

# Eigenvalues (/s2) [dof by 1 vector]
eig_val    = eigvals(Array(M\K))
# Circular frequency (/s) [dof by 1 vector]
omega      = sort(sqrt.(eig_val))
# Coefficient of Rayleigh damping (s)
dmp_b      = 2 * (zeta[1]*omega[1] - zeta[2]*omega[2]) / (omega[1]^2 -omega[2]^2)
dmp_a      = 2 * zeta[1]*omega[1]  - dmp_b*omega[1]^2
# Rayleigh damping (kg/s) [dof by dof matrix]
C          = dmp_a * M + dmp_b * K

# Parameters of Bouc-Wen model
const alpha  = 0.04                   # Ratio of linear to nonlinear response
const A      = 1                      # Basic hysteresis shape controls of Bouc-Wen model
const n      = 1                	  # Sharpness of yield of Bouc-Wen model
const beta   = 15         	          # Basic hysteresis shape controls of Bouc-Wen model (/m)
const gamma  = 150         	          # Basic hysteresis shape controls of Bouc-Wen model (/m)
const d_nu   = 1000        	          # Strength degradation of Bouc-Wen model (/m2)
const d_eta  = 1000                   # Stiffness degradation of Bouc-Wen model (/m2)
const p      = 1000            	      # Pinching slope of Bouc-Wen model (/m2)
const q      = 0.25             	  # Pinching initiation of Bouc-Wen model
const d_psi  = 5           	          # Pinching rate of Bouc-Wen model (/m2)
const lambda = 0.5        	          # Pinching severity/rate interaction of Bouc-Wen model
const zeta_s = 0.99                   # Measure of total slip of Bouc-Wen model
const psi    = 0.05                   # Pinching magnitude of Bouc-Wen model (m)

## Parameters of excitations
#--------------------------------------------------------
T       = 10.0                        # Time domain (s)
dt      = 0.005                       # Time step (s)
tspan   = collect(0.0:dt:T)           # time instants
nt      = length(tspan)               # number of time steps
D       = 0.0072                      # Intensity of Gaussian white noise (m2/s3)
# Load position [4*dof by 1 vector]
g       = [zeros(Float64, n_DoFs); ones(Float64, n_DoFs); zeros(Float64, 2*n_DoFs)]

## Pre-allocations
#--------------------------------------------------------
# Bounc-Wen model
df      = zeros(Float64, 4*n_DoFs)
x1      = zeros(Float64, n_DoFs)
x2      = zeros(Float64, n_DoFs)
x3      = zeros(Float64, n_DoFs)
x4      = zeros(Float64, n_DoFs)
dx1     = zeros(Float64, n_DoFs)
dx2     = zeros(Float64, n_DoFs)
nu      = zeros(Float64, n_DoFs)
eta     = zeros(Float64, n_DoFs)
zeta1   = zeros(Float64, n_DoFs)
zeta2   = zeros(Float64, n_DoFs)
zu      = zeros(Float64, n_DoFs)
h       = zeros(Float64, n_DoFs)
G0      = zeros(Float64, n_DoFs)
G       = zeros(Float64, n_DoFs)
bwpar   = BWPar( M, C, str_k, df, n_DoFs, alpha, A, n, beta, gamma, d_nu, d_eta, p, q, d_psi,
                 lambda, zeta_s, psi, x1, x2, x3, x4, dx1, dx2, nu, eta, zeta1, zeta2,
                 zu, h, G0, G, Minv)

##########################################################################################
## Structural analyses at samples
##########################################################################################
# Number of representative samples
N         = 800

# Initial values of equation of motion [4*dof by 1 vector]
ini_val   = zeros(Float64, 4*n_DoFs)

# preallocation for solutions
#--------------------------------------------------------
sol_eom    = zeros(Float64, 4*n_DoFs, nt)
disp_samp  = zeros(Float64, N, nt)
velo_samp  = zeros(Float64, N, nt)
drift      = zeros(Float64, n_DoFs, nt)
nforce     = zeros(Float64, n_DoFs, nt)
nforce_is  = zeros(Float64, n_DoFs, nt)
sforce     = zeros(Float64, n_DoFs, nt)
nforce_samp = zeros(Float64, N, nt)

## solve the equation of motion
#--------------------------------------------------------
@inbounds for i in 1:N
    srk2!(boucwen, g, bwpar, ini_val, D, dt, nt, sol_eom)
    @views disp_samp[i,:]     .= sol_eom[iDoF, :]
    @views velo_samp[i,:]     .= sol_eom[n_DoFs+iDoF, :]

    @views drift[1, :]        .= sol_eom[1, :]
    @views drift[2:n_DoFs, :] .= sol_eom[2:n_DoFs,:] .- sol_eom[1:n_DoFs-1,:]
    @views nforce .= str_k .* (alpha .* drift .+ (1-alpha) .* sol_eom[2*n_DoFs+1:3*n_DoFs,:])
    @views nforce_is[1:n_DoFs-1, :]  .= nforce[1:n_DoFs-1,:] - nforce[2:n_DoFs,:]
    @views nforce_is[n_DoFs, :]      .= nforce[n_DoFs,:]
    @views sforce .= -Minv * (C * sol_eom[n_DoFs+1:2*n_DoFs, :] .+ nforce_is)
    @views nforce_samp[i,:]   .= sforce[iDoF, :]

    if mod(i,100) == 0
        @printf("Evaluating the response of the %d-th/%d samples.\n", i, N)
    end
end

##########################################################################################
# Identification of intrinsic drift coefficients
##########################################################################################
# number of mesh grids along X-direction for IDC
n_xfit = 41
# number of mesh grids along V-direction for IDC
n_vfit = 41
# boundary of X response
b_x    = 0.5 # m
# boundary of V response
b_v    = 0.5 #m/s

# smoothing parameters for LOWESS
ptsfrac        = 0.2
n_pts          = N

# number of points for local regression
const n_locpts = round(Int64, ptsfrac*n_pts)

# mesh grids for IDC
knots     = gridknots([-b_x; -b_v], [b_x; b_v], [n_xfit; n_vfit])
xl_crude  = -b_x : (b_x*2.0/(n_xfit-1)) : b_x
vl_crude  = -b_v : (b_v*2.0/(n_vfit-1)) : b_v

# number of grid points for IDC
n_knots   = size(knots)[2]

# intermediate variables
idc     = zeros(Float64, n_knots)
weights = zeros(Float64, 2*n_locpts)
b       = @MVector zeros(Float64, 3)
stdvs   = @MVector zeros(Float64, 2)
locvals = zeros(Float64, 2*n_locpts)
re_coeff= @MVector zeros(Float64, 3)

locpts  = zeros(Float64, 2, 2*n_locpts)
lb_tmp  = @MVector zeros(Float64, 3)
AA      = @MMatrix zeros(Float64, 3, 3)
pts_n   = zeros(Float64, 2, 2*n_pts)
knots_n = zeros(Float64, 2, n_knots)

#! the number of columns is 'nt-1' rather than 'nt'
idc_ts  = zeros(Float64, n_knots, nt-1)
#! the number of points and function values used for LOWESS is '2*N' rather than 'N', 
#!      since the response is symmetrical.
pt_sym  = zeros(Float64, 2, 2*N)
f_sym   = zeros(Float64, 2*N)

# instance of composite type for LOWESS
lowess_par = lowessPar(n_knots, idc, weights, b, stdvs, locvals, re_coeff, locpts, lb_tmp, AA, pts_n, knots_n)
# instances of composite type for fitting IDC
idc_input  = idcInput(disp_samp, velo_samp, nforce_samp, knots, n_pts, nt)
idc_temp   = idcTemp(pt_sym, f_sym, idc_ts)

# evaluate the intrinsic drift coefficient
idc_ts     = idcfit!(idc_input, n_locpts, lowess_par, idc_temp)

##########################################################################################
# Solution by GE-GDEE
##########################################################################################
# number of mesh grids along X-direction for PDF
n_xl   = 201
# number of mesh grids along V-direction for PDF
n_vl   = 201
# mesh size for PDF
dxl    = 2*b_x/(n_xl-1)
dvl    = 2*b_v/(n_vl-1)
# mesh information for path integral method
xl     = collect(-b_x : dxl : b_x)
vl     = collect(-b_v : dvl : b_v)
xm     = reduce(vcat, xl' for i in 1:n_vl)
vm     = reduce(hcat, vl for i in 1:n_xl)
xmh    = xm .- vm .* dt

# intermediate variables
xlex       = zeros(Float64, n_xl+2)
pchipfvals = zeros(Float64, n_xl+2)
pdfmath    = zeros(Float64, n_vl, n_xl)
idcmath    = zeros(Float64, n_vl, n_xl)
tpdmat     = zeros(Float64, n_vl, n_vl)
mean_tpd_xi = zeros(Float64, n_vl)

# initial confition
pdfmat_ini = zeros(Float64, n_vl, n_xl)
pdfmat_ini[Int((n_vl-1)/2)+1, Int((n_xl-1)/2)+1] = 1.0 / dxl / dvl
pdfmat     = deepcopy(pdfmat_ini)

pdf_x      = zeros(Float64, n_xl, nt)
pdf_v      = zeros(Float64, n_vl, nt)
# idc_crude  = zeros(Float64, n_xfit, n_vfit)
idcmat     = zeros(Float64, n_vl, n_xl)

# composite-type input data for path integral
pis_par     = PISPar(n_xl, n_vl, dxl, dvl, xl, vl, xm, vm, xlex, pchipfvals, 
                     xmh, pdfmath, idcmath, tpdmat, mean_tpd_xi)

gegdee_temp = gegdeeTemp(idcmat, pdfmat, pdf_x, pdf_v)

idc_info    = idcInfo(nt, dt, D, n_xfit, n_vfit, xl_crude, vl_crude)

# solve the GE-GDEE
(pdf_x, pdf_v)   = gegdeesolver!(idc_ts, idc_info, pdfmat_ini, pis_par, gegdee_temp)

##########################################################################################
# Solution by Monte Carlo simulation
##########################################################################################
# number of samples for Monte Carlo simulation
n_mcs = Int(1e5)
# preallocation for solutions
sol_eom     = zeros(Float64, 4*n_DoFs, nt)
disp_samp1  = zeros(Float64, n_mcs, nt)
velo_samp1  = zeros(Float64, n_mcs, nt)

# solve the equation of motion
@inbounds for i in 1:n_mcs
    srk2!(boucwen, g, bwpar, ini_val, D, dt, nt, sol_eom)
    @views disp_samp1[i,:]     .= sol_eom[iDoF, :]

    if mod(i,1000) == 0
        @printf("Monte Carlo simulation processed: %7.2f %%\n", i/n_mcs*100)
    end
end

##########################################################################################
## evaluate cumulative distribution function (CDF) and statistical moments
##########################################################################################
# GE-GDEE / DR-PDEE
(cdf_x, cdf_v)   = cdfgegdee(pdf_x, pdf_v, dxl, dvl)                            # cumulative distribution function
(mean_x, mean_v) = meangegdee(pdf_x, pdf_v, xl, vl, dxl, dvl)                   # mean value process
(var_x, var_v)   = vargegdee(pdf_x, pdf_v, mean_x, mean_v, xl, vl, dxl, dvl)    # variance process
stdv_x = sqrt.(var_x);
stdv_v = sqrt.(var_v);                                                          # Standard deviation
# (skew_x, skew_v) = skewgegdee(pdf_x, pdf_v, mean_x, mean_v, stdv_x, stdv_v, xl, vl, dxl, dvl) # skewness process
# (kurt_x, kurt_v) = kurtgegdee(pdf_x, pdf_v, mean_x, mean_v, stdv_x, stdv_v, xl, vl, dxl, dvl) # kurtosis process

# Monte Carlo simulation
ecdf_mcs   = mcsecdfs(disp_samp1, xl)                                             # cumulative distribution function
mean_x_mcs = sum(disp_samp1, dims = 1) / n_mcs                                  # mean value process
var_x_mcs  = sum((disp_samp1 .- mean_x_mcs) .^ 2, dims = 1) ./ (n_mcs-1)        # variance process
stdv_x_mcs = sqrt.(var_x_mcs)                                                   # Standard deviation
# skew_x_mcs = sum(((disp_samp1 .- mean_x_mcs) ./ stdv_x_mcs)  .^ 3, dims = 1) ./ n_mcs  # skewness process
# kurt_x_mcs = sum(((disp_samp1 .- mean_x_mcs) ./ stdv_x_mcs)  .^ 4, dims = 1) ./ n_mcs  # kurtosis process

##########################################################################################
## Visualization
##########################################################################################
# Standard deviation
plot(tspan, stdv_x', ls=:solid, lw=1.5, label="DR-PDEE")
plot!(tspan, stdv_x_mcs', ls=:dash, lw=2.0, label="Monte Carlo simulation")
xlabel!("Time [s]")
ylabel!("Standard deviation [m]")

# Probability density function
histogram(disp_samp1[:,end],bins = range(-0.5, 0.5, length=61),
            normalize=:pdf, color=:gray, label="Monte Carlo simulation")
plot!(xl, pdf_x[:,end], ls=:solid, lw=2.0, label="DR-PDEE")
xlabel!("Displacement [m]")
ylabel!("Probability density function")

# Cumulative distribution function
plot(xl, cdf_x[:,end].+1e-10, yscale=:log10, ylims=(1e-6, 1), ls=:solid, lw=2.0, label="DR-PDEE")
plot!(xl, ecdf_mcs[:,end].+1e-10, yscale=:log10, ylims=(1e-6, 1), ls=:dash, lw=2.0, label="Monte Carlo simulation")
xlabel!("Displacement [m]")
ylabel!("Cumulative distribution function")