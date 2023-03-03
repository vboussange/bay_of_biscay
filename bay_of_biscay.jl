#=
This Julia code generates time series from a 5-compartment 
ecosystem model described in https://www.jstor.org/stable/177129
and performs inference on the model. 

author: Victor Boussange
email: vic.boussange@gmail.com
=#

#= 
The `Graphs`, and `SparseArrays` packages are used to define the foodweb and
manipulate related parameters. The `EcoEvoModelZoo` package provides access to a
collection of ecosystem models, and the `ParametricModels` package is used to
define and manipulate models with parameters. The `OrdinaryDiffEq` package
provides tools for solving ordinary differential equations, while the
`LinearAlgebra` package is used for linear algebraic computations. The `UnPack`
package provides a convenient way to extract fields from structures, and the
`ComponentArrays` package is used to store and manipulate the model parameters
conveniently. Finally, the `PythonCall` package is used to interface with
Python's Matplotlib library for visualization.
=#
using Graphs
using EcoEvoModelZoo
using ParametricModels
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics
using SparseArrays
using ComponentArrays
using PythonPlot
# plt = pyimport("matplotlib.pyplot");

#=
Defining hyperparameters for the forward simulation of the model.

The `alg` variable specifies the algorithm used for solving the differential equations,
while `abstol` and `reltol` specify the absolute and relative tolerances for the
solver. `tspan` specifies the time interval for the simulation, and `tsteps`
specifies the time steps at which to save the output.
=#
alg = BS3()
abstol = 1e-6
reltol = 1e-6
tspan = (0.0, 600)
tsteps = range(300, tspan[end], length=100)

#=
Defining ecosystem model

Next, the ecosystem model is defined. We first build the `foodweb`.
The `N` variable specifies the number of
compartments in the model. The `add_edge!` function is used to add edges to the
graph, specifying the flow of resources between compartments.
=#
N = 5 # number of compartment

foodweb = DiGraph(N)
add_edge!(foodweb, 2 => 1) # C1 to R1
add_edge!(foodweb, 5 => 4) # C2 to R2
add_edge!(foodweb, 3 => 2) # P to C1
add_edge!(foodweb, 3 => 5) # P to C2

I, J, _ = findnz(adjacency_matrix(foodweb))

#=
For fun, let's just plot the 
foodweb.
=#
using PythonCall
nx = pyimport("networkx")
np = pyimport("numpy")
plt = pyimport("matplotlib.pyplot")
species_colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]

g_nx = nx.DiGraph(np.array(adjacency_matrix(foodweb)))
pos = Dict(0 => [0, 0], 1 => [1, 1], 2 => [2, 2], 3 => [4, 0], 4 => [3, 1])

fig, axs = plt.subplots(1,2, figsize=(10,4))
nx.draw(g_nx, pos, ax=axs[1], node_color=species_colors, node_size=1000)
display(fig)

#=
The next several functions are required by `SimpleEcosystemModel` and define the
specific dynamics of the model. The `intinsic_growth_rate` function specifies the
intrinsic growth rate of each compartment, while the `carrying_capacity` function
specifies the carrying capacity of each compartment. The `competition` function
specifies the competition between and within compartments, while the
`resource_conversion_efficiency` function specifies the efficiency with which
resources are converted into consumer biomass. The `feeding` function specifies
the feeding interactions between compartments.
=#
intinsic_growth_rate(p, t) = p.r

function carrying_capacity(p, t)
    @unpack K₁₁ = p
    K = vcat(K₁₁, ones(N - 1))
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
    W = sparse(I, J, vcat(1.0, ω, 1.0, 1 .- ω))

    # handling time
    H = sparse(I, J, vcat(H₂₁, H₃₂, H₅₄, H₃₅))

    # attack rates
    q = sparse(I, J, vcat(q₂₁, q₃₂, q₅₄, q₃₅))

    return q .* W ./ (one(eltype(u)) .+ q .* H .* (W * u))
end

#=
Defining ecosystem model parameters

The parameters for the ecosystem model are defined using a `ComponentArray`. The
u0_true variable specifies the initial conditions for the simulation. The
ModelParams function from the ParametricModels package is used to specify the
model parameters and simulation settings. Finally, the SimpleEcosystemModel
function from the EcoEvoModelZoo package is used to define the ecosystem model.
=#

p_true = ComponentArray(ω=[0.2],
    H₂₁=[2.89855],
    H₅₄=[2.89855],
    H₃₂=[7.35294],
    H₃₅=[7.35294],
    q₂₁=[1.38],
    q₅₄=[1.38],
    q₃₂=[0.272],
    q₃₅=[0.272],
    r=[1.0, -0.15, -0.08, 1.0, -0.15],
    K₁₁=[1.0],
    A₁₁=[1.0],
    A₄₄=[1.0])

u0_true = [0.77, 0.060, 0.945, 0.467, 0.18]


mp = ModelParams(; p=p_true,
    tspan,
    u0=u0_true,
    alg,
    reltol,
    abstol,
    saveat=tsteps,
    verbose=false, # suppresses warnings for maxiters
    maxiters=50_000
)
model = SimpleEcosystemModel(; mp, intinsic_growth_rate,
    carrying_capacity,
    competition,
    resource_conversion_efficiency,
    feeding)
data = simulate(model, u0=u0_true) |> Array
data = data .* exp.(0.1 * randn(size(data)))

# plotting
using PythonCall;
plt = pyimport("matplotlib.pyplot");
ax = axs[0]
for i in 1:N
    ax.plot(data[i, :], label="Species $i", color = species_colors[i])
end
# ax.set_yscale("log")
ax.set_ylabel("Species abundance")
ax.set_xlabel("Time (days)")
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]
fig.legend()
fig.savefig("time_series_5_species_ecosyste_model.png")
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

loss_likelihood(data, pred, rg) = sum((data .- pred) .^ 2)# loss_fn_lognormal_distrib(data, pred, noise_distrib)
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
    push!(θs, err)
    if plotting && length(losses) % info_per_its == 0
        # print_param_values(re(p_trained), p_true)
        plotting_fit(losses, pred, ranges, data, tsteps)
    end
end


stats = @timed piecewise_MLE(infprob;
    adtype,
    group_size=group_size,
    batchsizes=batchsizes,
    data=data,
    tsteps=tsteps,
    optimizers=optimizers,
    epochs=epochs,
    verbose_loss=verbose,
    info_per_its=info_per_its,
    cb=callback
)