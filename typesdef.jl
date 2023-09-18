# This file defines composite types

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

# define a composite type for the Bouc-Wen model (in function: boucwen)
# #######################################################################
struct BWPar
    # structural matrices
    #----------------------------------
    M::Matrix{Float64}          # mass matrix
    C::Matrix{Float64}          # damping matrix
    k0::Vector{Float64}         # initial lateral stiffness vector
    df::Vector{Float64}         # right hand term of the equation of motion (expect for excitation)
    n_DoFs::Int64               # number of degrees of freedom

    # parameters of Bouc-Wen model
    #----------------------------------
    alpha::Float64
    A::Float64
    n::Float64
    beta::Float64
    gamma::Float64
    d_nu::Float64 
    d_eta::Float64    
    p::Float64
    q::Float64       
    d_psi::Float64
    lambda::Float64
    zeta_s::Float64
    psi::Float64

    # intermediate variables
    #----------------------------------
    x1::Vector{Float64}
    x2::Vector{Float64} 
    x3::Vector{Float64}
    x4::Vector{Float64}  
    dx1::Vector{Float64}
    dx2::Vector{Float64}
    nu::Vector{Float64}
    eta::Vector{Float64}
    zeta1::Vector{Float64}
    zeta2::Vector{Float64}
    zu::Vector{Float64}
    h::Vector{Float64}
    G0::Vector{Float64}
    G::Vector{Float64}
    Minv::Matrix{Float64}       # inverse of the mass matrix

end

# define a composite type including pre-allocated intermediate variables in
#       the determination of the intrinsic drift coefficient. (in function: idcfit!)
# #######################################################################
struct idcTemp
    pts    :: Matrix{Float64}
    fs     :: Vector{Float64}
    idc_ts :: Matrix{Float64}
end

# define a composite type including necessary information in
#       the determination of the intrinsic drift coefficient. (in function: idcfit!)
# #######################################################################
struct idcInput
    disp_samp   :: Matrix{Float64}  # displacement response (each column associating with a sample)
    velo_samp   :: Matrix{Float64}  # velocity response (each column associating with a sample)
    nforce_samp :: Matrix{Float64}  # force response (each column associating with a sample)
    knots       :: Matrix{Float64}  # points at which the intrinsic drift coefficients is to be fitted
    n_samp      :: Int64            # number of representative samples
    nt          :: Int64            # number of time steps
end

# define a composite type including necessary information and pre-allocated intermediate
#        variables in LOWESS. (in functions: idcfit! & idclowess)
# #######################################################################
struct lowessPar
    # number of mesh grids
    n_knots :: Int64

    # output: intrinsic drift coefficients
    idc     :: Vector{Float64}

    # intermediate variables (Vector)
    weights :: Vector{Float64}
    b       :: MVector{3,Float64}
    stdvs   :: MVector{2,Float64}
    locvals :: Vector{Float64}
    re_coeff:: MVector{3,Float64}

    # intermediate variables (Matrix)
    locpts  :: Matrix{Float64}
    lb_tmp  :: MVector{3,Float64}
    A       :: MMatrix{3,3,Float64}
    pts_n   :: Matrix{Float64}
    knots_n :: Matrix{Float64}
end

# define a composite type including necessary information and pre-allocated intermediate
#       variables for path integral method. (in functions: gegdeesolver! & pathintegral!)
# #######################################################################
struct PISPar
    n_xfine    :: Int64             # number of grids along x-direction (refined mesh)
    n_vfine    :: Int64             # number of girds along v-direction (refined mesh)
    dxl_fine   :: Float64           # mesh size along x-direction
    dvl_fine   :: Float64           # mesh size along v-direction

    xl_fine    :: Vector{Float64}   # range of grids along x-direction  (refined mesh)
    vl_fine    :: Vector{Float64}   # range of grids along v-direction  (refined mesh)
    xm_fine    :: Matrix{Float64}   # matrix of fine girds coordinates along x-direction
    vm_fine    :: Matrix{Float64}   # matrix of fine girds coordinates along v-direction

    xlex :: Vector{Float64}       # extended mesh grids along the X-direction for interpolation
    pchipfvals :: Vector{Float64} # extended function values along the X-direction for interpolation

    xmh  :: Matrix{Float64}       # mesh grids for PDF(xi-vj*dt, vj, th-1)

    pdfmath :: Matrix{Float64}    # matrix of PDF(xi-vj*dt, vj, th-1)
    idcmath :: Matrix{Float64}    # matrix of IDC(xi-vj*dt, vj, th-1)
    tpdmat  :: Matrix{Float64}    # matrix of transition probability density (TPD)
    mean_tpd_xi :: Vector{Float64}
end

# define a composite type to transfer all information of intrinsic drift coefficient
#       for solving the GE-GDEE/DR-PDEE. (in functions: gegdeesolver! & pathintegral!)
# #######################################################################
struct idcInfo
    n_step     ::Int64              # number of time steps
    dt         :: Float64           # time step size
    D          :: Float64           # intensity of gaussian white noise
    n_xcrude   :: Int64             # number of grids along x-direction (crude mesh)
    n_vcrude   :: Int64             # number of girds along v-direction (crude mesh)
    xl_crude   :: StepRangeLen      # range of grids along x-direction  (crude mesh)
    vl_crude   :: StepRangeLen      # range of grids along v-direction  (crude mesh)
end

# define a composite types including pre-allocated intermediate variables for
#       for solving the GE-GDEE/DR-PDEE. (in function: gegdeesolver!)
# #######################################################################
struct gegdeeTemp
    idcmat :: Matrix{Float64}       # matrix of intrinsic drift coefficient
    pdfmat :: Matrix{Float64}       # matrix of joint probability density function
    pdf_x  :: Matrix{Float64}       # matrix of probability density function of displacement
    pdf_v  :: Matrix{Float64}       # matrix of probability density function of velocity
end
