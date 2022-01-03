# __precompile__(false)

module TNLatent

using Base: register_root_module, process_events
include("./utils.jl")
include("./type.jl")
include("./synthetic.jl")

include("./tensor_basics.jl")
include("./tn_tools.jl")
include("./tn_latent.jl")
include("./trlf_em.jl")
include("./trlf_em_approx.jl")
include("./trlf_em_completion.jl")
include("./trlf_gibbs.jl")

export genMask, image2Array, array2Image, array2Patch, patch2Array
export RSE, MyPSNR
export band, ped, lin_band, lin_ped
export Optim, TRLF
export matricization, unfolding, un_matricization, folding, fullCores
export init!, fromList!, fullData, computeCoef, computeΣ
export TRLF_EM, TRLF_GS


end
