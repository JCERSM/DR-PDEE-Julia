# This file defines a module for the solution of the DR-PDEE/GE-GDEE

# By Yang Jiashu @ Tongji University
# Email: jiashuyang@tongji.edu.cn

module DRPDEE

    # Export functions
    export srk2!, idcfit!, gegdeesolver!, cdfgegdee, meangegdee, vargegdee, 
            mcsecdfs, boucwen, gridknots, idclowess, pathintegral!

    export BWPar, idcTemp, idcInput, lowessPar, PISPar, idcInfo, gegdeeTemp

    # Required packages
    using Distributions
    using Interpolations
    using LinearAlgebra
    using NearestNeighbors
    using Printf
    using Random
    using Statistics
    using StaticArrays

    #  Source code files
    include("./typesdef.jl")
    include("./gridknots.jl")
    include("./srk2.jl")
    include("./boucwen.jl")
    include("./idclowess.jl")
    include("./pathintegral.jl")
    include("./gegdeesolver.jl")
    include("./idcfit.jl")
    include("./gegdeepostproc.jl")

end