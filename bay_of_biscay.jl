#=
Generation of time series from 
a 5 compartment ecosystem model,
and inference.

Author: Victor Boussange, vic.boussang@gmail.com
=#
using SimpleWeightedGraphs
using EcoEvoModelZoo
using ParametricModels
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics
using Graphs
using SparseArrays
using ComponentArrays
using PythonCall; plt = pyimport("matplotlib.pyplot")

# https://www.jstor.org/stable/177129

#=
Defining hyperparameters
=#
alg = BS3()
abstol = 1e-6
reltol = 1e-6
tspan = (0., 600)
tsteps = range(300, tspan[end], length=100)

#=
Defining ecosystem model
=#
N = 5 # number of compartment

foodweb = DiGraph(N)
add_edge!(foodweb, 2 => 1) # C1 to R1
add_edge!(foodweb, 5 => 4) # C2 to R2
add_edge!(foodweb, 3 => 2) # P to C1
add_edge!(foodweb, 3 => 5) # P to C2

I, J, _ = findnz(adjacency_matrix(foodweb))

intinsic_growth_rate(p, t) = p.r

function carrying_capacity(p, t)
    @unpack K₁₁ = p
    K = vcat(K₁₁, ones(N-1))
    return K
end

function competition(u, p, t)
    @unpack A₁₁, A₄₄ = p
    A = spdiagm(vcat(A₁₁, 0, 0, A₄₄, 0))
    return A * u
end

resource_conversion_efficiency(p, t) = ones(N)

function feeding(u, p, t)
    @unpack ω, H₂₁, H₅₄, H₃₂, H₃₅, q₂₁, q₅₄, q₃₂, q₃₅ = p

    # creating foodweb
    W = sparse(I, J, vcat(1., ω, 1., 1 .- ω))

    # handling time
    H = sparse(I, J, vcat(H₂₁, H₃₂, H₅₄, H₃₅))
    
    # attack rates
    q = sparse(I, J, vcat(q₂₁, q₃₂, q₅₄, q₃₅))
    
    return q .* W ./ (one(eltype(u)) .+ q .* H .* (W * u))
end

#=
Defining ecosystem model parameters
=#

p_true = ComponentArray(ω = [0.2], 
                        H₂₁ = [2.89855], 
                        H₅₄ = [2.89855], 
                        H₃₂ = [7.35294], 
                        H₃₅ = [7.35294], 
                        q₂₁ = [1.38], 
                        q₅₄ = [1.38], 
                        q₃₂ = [0.272], 
                        q₃₅ = [0.272], 
                        r = [1.0, -0.15, -0.08, 1.0, -0.15], 
                        K₁₁ = [1.],
                        A₁₁ = [1.],
                        A₄₄ = [1.])

u0_true = [0.77, 0.060, 0.945, 0.467, 0.18]


mp = ModelParams(;p = p_true,
                tspan,
                u0 = u0_true,
                alg,
                reltol,
                abstol,
                saveat = tsteps,
                verbose = false, # suppresses warnings for maxiters
                maxiters = 50_000,
                )
model = SimpleEcosystemModel(;mp, intinsic_growth_rate, 
                                carrying_capacity, 
                                competition, 
                                resource_conversion_efficiency, 
                                feeding)
data = simulate(model, u0 = u0_true) |> Array
data = data .* exp.(0.1 * randn(size(data)))

# plotting
using PythonCall; plt = pyimport("matplotlib.pyplot")
fig, ax = plt.subplots(1)
for i in 1:N
    ax.plot(data[i,:], label = "Species $i")
end
# ax.set_yscale("log")
fig.legend()
display(fig)



####################
#### Inference #####
####################
using PiecewiseInference
include("cb.jl") # to monitor losses and fit
include("loss.jl") # provides log loss function
using OptimizationFlux
using SciMLSensitivity

# hyperparameters for the inference process
# adtype = Optimization.AutoForwardDiff()
adtype = Optimization.AutoZygote()
optimizers = [Adam(1e-2)]
epochs = [1000]
batchsizes = [3]
group_size = 10 + 1 # number of points in each segment
verbose = true
info_per_its = 1
plotting = true

p_init = p_true
p_init.ω .= 0.1 #

loss_likelihood(data, pred, rg) = sum((data .- pred).^2)# loss_fn_lognormal_distrib(data, pred, noise_distrib)
# loss_u0_prior(u0_data, u0_pred) = loss_fn_lognormal_distrib(u0_data, u0_pred, u0_distrib)
# loss_param_prior(p) = loss_param_prior_from_dict(p, prior_param_distrib)

infprob = InferenceProblem(model, 
                            p_init; 
                            # loss_param_prior, 
                            # loss_u0_prior,
                            loss_likelihood
                            )

θs = []
function callback(p_trained, losses, pred, ranges)
    err = median([median(abs.((p_trained[k] - p_true[k]) ./ p_true[k])) for k in keys(p_true)])
    push!(θs,err)
    if plotting && length(losses) % info_per_its == 0
        # print_param_values(re(p_trained), p_true)
        plotting_fit(losses, pred, ranges, data, tsteps)
    end
end


stats = @timed piecewise_MLE(infprob;
                            adtype,
                            group_size = group_size,
                            batchsizes = batchsizes,
                            data = data, 
                            tsteps = tsteps,
                            optimizers = optimizers,
                            epochs = epochs,
                            verbose_loss = verbose,
                            info_per_its = info_per_its,
                            cb = callback,
                            )