using Distributions

function loss_fn_lognormal_distrib(data, pred, noise_distrib)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
        @show "hello"
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
        @show "hello"

    end

    l = 0.

    # observations
    ϵ = log.(data) .- log.(pred)
    for i in 1:size(ϵ, 2)
        l += logpdf(noise_distrib, ϵ[:, i])
    end
    # l /= size(ϵ, 2) # TODO: this bit is unclear

    if l isa Number # preventing any other reason for Zygote to crash
        return - l
    else 
        return Inf
    end
end