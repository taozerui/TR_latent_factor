using Pkg

dependencies = [
    "IJulia",
    "TensorOperations",
    "OMEinsum",
    "Einsum",
    "Distributions",
    "EllipsisNotation",
    "BenchmarkTools",
    "Plots",
    "Parameters",
    "PosDefManifold",
    "CovarianceEstimation",
    "Images",
    "Random",
    "LinearAlgebra"
]

Pkg.add(dependencies)
