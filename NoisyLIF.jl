using DifferentialEquations
using Plots
using Random
using LsqFit

global seed = 1234
Random.seed!(seed)

function lif(u, p, t)
    gL, EL, C, Vth, I, noise_std = p
    I_noisy = I + noise_std * randn()
    (-gL * (u - EL) + I_noisy) / C
end

function thr(u, t, integrator)
    integrator.u > integrator.p[4]
end

function reset!(integrator)
    integrator.u = integrator.p[2]
    global spike_count += 1
end

u0 = -75
tspan = (0.0, 40.0)
# p = (gL, EL, C, Vth, I, noise_std)
p = [10.0, -75.0, 5.0, -55.0, 0.0, 5.0]

threshold = DiscreteCallback(thr, reset!)
cb = CallbackSet(threshold)

I_values = range(0.0, stop=600.0, length=100)
spiking_rates = []
Lif_sols = []

for I in I_values
    global spike_count = 0
    p[5] = I
    prob = ODEProblem(lif, u0, tspan, p, callback=cb)
    sol = solve(prob)
    push!(Lif_sols, sol)
    spiking_rate = spike_count / (tspan[2] - tspan[1])
    push!(spiking_rates, spiking_rate)
end

plot(I_values, spiking_rates, xlabel="Current (I)", ylabel="Spiking rate (Hz)", legend=false)

# Define the sigmoid function
sigmoid(x, p) = p[1] ./ (1 .+ exp.(-p[2] * (x .- p[3])))

# Define the x / (x + 1) function
function x_over_x_plus_1(x, p)
    output = p[1] .* (x .+ p[3]) ./ ((x .+ p[3]) .+ p[2])
    condition = ((x .+ p[3]) .+ p[2]) .<= 0
    return max.(output .* (.!condition), 0)
end

# Define the modified xx1_gain_cor function for fitting

function xx1_gain_cor_fit(x, p)
    x = convert(Array{Float64}, x)
    gain = p[1]
    gain_cor_range = p[2]
    gain_cor = p[3]
    n_var = p[4]
    translation = p[5]

    gain_cor_fact = (gain_cor_range .- ((x .+ translation) ./ n_var)) ./ gain_cor_range

    new_gain = gain .* (1 .- gain_cor .* gain_cor_fact)
    new_gain[gain_cor_fact .< 0] .= 1
    condition = (new_gain .* (x .+ translation) .+ 1) .<= 0
    output = max.(new_gain .* (x .+ translation) ./ (new_gain .* (x .+ translation) .+ 1), 0)
    return output #(output .* (.!condition))
end

# Perform the curve fit for sigmoid
p0_sigmoid = [1.0, 0.01, 100.0] # initial guess for sigmoid parameters
fit_sigmoid = curve_fit(sigmoid, I_values, spiking_rates, p0_sigmoid)

# Perform the curve fit for x / (x + 1)
p0_x_over_x_plus_1 = [1.0, 1.0, -100] # initial guess for x / (x + 1) parameters
fit_x_over_x_plus_1 = curve_fit(x_over_x_plus_1, I_values, spiking_rates, p0_x_over_x_plus_1)

# Perform the curve fit for xx1_gain_cor_fit

p0_xx1_gain_cor_fit = [100.0, 10.0, 10.0, 1.0, -200.0] # initial guess for xx1_gain_cor_fit parameters
fit_xx1_gain_cor_fit = curve_fit(xx1_gain_cor_fit, I_values, spiking_rates, p0_xx1_gain_cor_fit)

# Calculate the sum of squared errors (SSE) for all three fits
sse_sigmoid = sum(abs2, fit_sigmoid.resid)
sse_x_over_x_plus_1 = sum(abs2, fit_x_over_x_plus_1.resid)
sse_xx1_gain_cor_fit = sum(abs2, fit_xx1_gain_cor_fit.resid)

# Plot the data and the fitted curves
I_fitted = range(0.0, stop=600.0, length=1000)
spiking_rates_fitted_sigmoid = sigmoid(I_fitted, fit_sigmoid.param)
spiking_rates_fitted_x_over_x_plus_1 = x_over_x_plus_1(I_fitted, fit_x_over_x_plus_1.param)
spiking_rates_fitted_xx1_gain_cor_fit = xx1_gain_cor_fit(I_fitted, fit_xx1_gain_cor_fit.param)

fig = plot(I_values, spiking_rates, seriestype=:scatter, label="Data", xlabel="Current (I)", ylabel="Spiking rate (Hz)", legend=:topleft)
fig = plot!(I_fitted, spiking_rates_fitted_sigmoid, label="Fitted sigmoid", linewidth=2, linestyle=:dash, color=:blue)
fig = plot!(I_fitted, spiking_rates_fitted_x_over_x_plus_1, label="Fitted x / (x + 1)", linewidth=2, linestyle=:dash, color=:red)
#fig = plot!(I_fitted, spiking_rates_fitted_xx1_gain_cor_fit, label="Fitted xx1_gain_cor", linewidth=2, linestyle=:dash, color=:green)


# Plot functions
x_values = range(-500, stop=500, length=1000)
noisy_xx1_values = [x_over_x_plus_1(x_values, fit_x_over_x_plus_1.param) for x in x_values]
xx1plot = plot(x_values, noisy_xx1_values, xlabel="x", ylabel="Smoothed XX1", legend=false)

noisy_xx1_values = [xx1_gain_cor_fit(x_values, fit_xx1_gain_cor_fit.param) for x in x_values]
xx1_gain_corplot = plot(x_values, noisy_xx1_values, xlabel="x", ylabel="Smoothed XX1", legend=false)

fig