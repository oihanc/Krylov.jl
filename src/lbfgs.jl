export lbfgs, lbfgs!
"""
    (x, stats) = lbfgs(A, b::AbstractVector{FC};
                       M=I, ldiv::Bool=false,
                       radius::T = zero(T),
                       atol::T = √eps(T),
                       rtol::T = √eps(T),
                       itmax::Int = 0,
                       timemax::Float64 = Inf,
                       verbose::Int = 0,
                       history::Bool = false,
                       callback = workspace -> false,
                       iostream::IO = kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = lbfgs(A, b, x0::AbstractVector; kwargs...)

L-BFGS can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

The limited-memory BFGS method is a limited-memory quasi-Newton method from the Broyden class,
used to solve smooth unconstrained optimization problems. The following implementation considers the strictly convex quadratic case

    f(x) = 1/2 xᵀ A x − bᵀ x,

with exact line search.

Note that the implementation does not support a preconditioner.

#### Interface

To easily switch between methods, use the generic interface [`krylov_solve`](@ref)
with `method = :lbfgs`.

#### Input arguments

* `A`: a linear operator that models the hessian operator of dimension `n`;
* `b`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments
* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `radius`: add the trust-region constraint ‖x‖ ≤ `radius` if `radius > 0`. Useful to compute a step in a trust-region method for optimization.
* `atol`: absolute stopping tolerance based on the gradient norm;
* `rtol`: relative stopping tolerance based on the gradient norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as gradient norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`DiomCgStats`](@ref) structure.

#### Reference

* Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical programming, 45(1), 503-528.
"""
function lbfgs end

"""
    workspace = lbfgs!(workspace::LbfgsWorkspace, A, b; kwargs...)
    workspace = lbfgs!(workspace::LbfgsWorkspace, A, b, x0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`lbfgs`](@ref).

See [`LbfgsWorkspace`](@ref) for instructions on how to create the `workspace`.

"""
function lbfgs! end

def_args_lbfgs = (:(A                    ),
                  :(b::AbstractVector{FC}))

def_optargs_lbfgs = (:(x0::AbstractVector),)

def_kwargs_lbfgs = (:(; M = I                            ),
                    :(; ldiv::Bool = false               ),
                    :(; radius::T = zero(T)              ),
                    :(; atol::T = √eps(T)                ),
                    :(; rtol::T = √eps(T)                ),
                    :(; itmax::Int = 0                   ),
                    :(; timemax::Float64 = Inf           ),
                    :(; verbose::Int = 0                 ),
                    :(; history::Bool = false            ),
                    :(; callback = workspace -> false    ),
                    :(; iostream::IO = kstdout           ))

def_kwargs_lbfgs = extract_parameters.(def_kwargs_lbfgs)

args_lbfgs = (:A, :b)
optargs_lbfgs = (:x0,)
kwargs_lbfgs = (:M, :ldiv, :radius, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function lbfgs!(workspace :: LbfgsWorkspace{T,FC,S}, $(def_args_lbfgs...); $(def_kwargs_lbfgs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "LBFGS: system of size %d\n", n)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    Δx, s, x, g, d, Ad, y = workspace.Δx, workspace.s, workspace.x, workspace.g, workspace.d, workspace.Ad, workspace.y
    warm_start = workspace.warm_start
    H, stats = workspace.H, workspace.stats

    LinearOperators.reset!(H)
    
    rNorms = stats.residuals
    qxs = stats.qvals
    reset!(stats)

    kfill!(x, zero(FC))  # x₀
    if warm_start
      mul!(g, A, Δx)
      (radius > 0) && (qx = kdot(n, Δx, g) / 2 - kdot(n, b, Δx))    # q(x₀) = ½ΔxᵀAΔx - bᵀΔx
      kaxpby!(n, -one(FC), b, one(FC), g)
    else
      kcopy!(n, g, -b)  # g ← -b
      (radius > 0) && (qx = zero(T))                 # q(0) = 0
    end

    kcopy!(n, d, -g) # d ← g
    γ = kdotr(n, g, d)
    γ ≤ 0 || error("The direction `d` is not a descent direction.")
    rNorm = sqrt(-γ)
    if history
      push!(rNorms, rNorm)
      (radius > 0) && push!(qxs, qx)
    end
    if γ == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    dAd = zero(T)
    dNorm² = -γ
    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖gₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    on_boundary = false
    zero_curvature = false
    user_requested_exit = false
    overtimed = false

    status = "unknown"

    while !(solved || tired || zero_curvature || user_requested_exit || overtimed)              
      mul!(Ad, A, d)        # compute Ad
      dAd = kdotr(n, d, Ad)  # compute curvature
      if (dAd ≤ eps(T) * dNorm²) && (radius == 0)
        if abs(dAd) ≤ eps(T) * dNorm²
          zero_curvature = true
        end
      end
      zero_curvature && continue
      
      α = -γ/dAd

      # Compute step size to boundary if applicable.
      if radius == 0
         σ = α
      else
         σ = maximum(to_boundary(n, x, d, -g, radius, dNorm2=dNorm²)) # compute step size to boundary
      end
      # Move along d from x to the boundary if either
      # the next step leads outside the trust region or
      # we have nonpositive curvature.

      if (radius > 0) && ((dAd ≤ 0) || (α > σ))
        α = σ
        if dAd ≤ 0
          stats.indefinite = true
        end
        on_boundary = true
      end

      s .= α .* d
      x .= x .+ s
      
      y .= α .* Ad
      g .= g .+ y

      push!(H, s, y)
      mul!(d, H, -g)        # compute search direction
      γ_next = kdotr(n, d, g)
      rNorm = knorm(n, g)
      γ_next ≤ 0 || error("The direction `d` is not a descent direction.")
      rNorm = sqrt(-γ_next)
      (radius > 0) && (qx += α^2 * dAd / 2 + α * γ) 

      if history
        radius > 0 && push!(qxs, qx)
        push!(rNorms, rNorm)
      end 

      # update stopping criteria
      user_requested_exit = callback(workspace) :: Bool
      
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))
      resid_decrease_lim = rNorm ≤ ε
      resid_decrease = resid_decrease_lim || resid_decrease_mach
      solved = resid_decrease || on_boundary
      
      if !solved
        β = γ_next / γ
        dNorm² = -γ_next + β^2 * dNorm²
        γ = γ_next  
      end
      

      iter += 1
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      
      overtimed = timer > timemax_ns
      
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)

    end

    # Termination status
    solved && on_boundary             && (status = "on trust-region boundary")
    solved && stats.indefinite        && (status = "nonpositive curvature detected")
    solved && (status == "unknown")   && (status = "solution good enough given atol and rtol")
    tired                             && (status = "maximum number of iterations exceeded")
    zero_curvature                    && (status = "zero curvature detected")
    user_requested_exit               && (status = "user-requested exit")
    overtimed                         && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
