

export lbfgs, lbfgs!



function lbfgs end


function lbfgs! end

def_args_lbfgs = (:(A                    ),
                  :(b::AbstractVector{FC}))

def_optargs_lbfgs = (:(x0::AbstractVector),)

def_kwargs_lbfgs = (:(; M = I                            ),
                    :(; N = I                            ),
                    :(; ldiv::Bool = false               ),
                    :(; radius::T = zero(T)              ),
                    :(; reorthogonalization::Bool = false),
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
kwargs_lbfgs = (:M, :N, :ldiv, :radius, :reorthogonalization, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function lbfgs!(workspace :: LbfgsWorkspace{T,FC,S}, $(def_args_lbfgs...); $(def_kwargs_lbfgs...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    b = b .* -1   # review implementation

    m, n = size(A)
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "LBFGS: system of size %d\n", n)

    # Check M = Iₙ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")

    s, p, g, d, Ad, y = workspace.s, workspace.x, workspace.g, workspace.d, workspace.Ad, workspace.y
    H, stats = workspace.H, workspace.stats

    LinearOperators.reset!(H)
    
    rNorms = stats.residuals
    qxs = stats.qvals
    reset!(stats)
    # w  = MisI ? t : workspace.w
    # r₀ = MisI ? t : workspace.w

    # Initial solution x₀ and residual r₀.
    kfill!(p, zero(FC))  # x₀
    
    # add warmstart
    
    kcopy!(n, g, b) # update initial gradient

    MisI || mulorldiv!(g, M, t, ldiv)  # M(b - Ax₀)
    rNorm = knorm(n, g)
    if history
      qx = zero(T)
      push!(qxs, qx)
      push!(rNorms, rNorm)
    end

    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), s, p)
      workspace.warm_start = false
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2*n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖gₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    on_boundary = false
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || user_requested_exit || overtimed)

      # Update iteration index.
      iter = iter + 1
      
      # push!(H, s, g)      # update Inverse BFGS operator
      
      mul!(d, -H, g)        # compute search direction
      mul!(Ad, A, d)        # compute Ad
      
      dAd = kdot(n, d, Ad)  # compute curvature
      gd = kdot(n, g, d)

      # handle negative curvature
      if radius > 0 && dAd <= 0
        alpha = -sign(gd)*2*radius/knorm(n, d)
      else
        alpha = -gd/dAd
      end

      # compute step size
      s .= alpha .* d
      p .= p .+ s
      
      # if step size is outside of radius -> project p on the radius
      if radius > 0 && knorm(n, p) > radius
        p .-= s
        # ps = kdot(n, p, s)
        # ss = kdot(n, s, s)
        # τ = (-ps + sqrt(ps^2 + ss*(radius^2 - kdot(n, p, p))))/ss
        
        τ = maximum(to_boundary(n, p, d, -g, radius))

        # kaxpy!(n, τ, s, p)    # p .= p .+ τ .* s
        kaxpy!(n, τ, d, p)    # p .= p .+ τ .* s
        
        on_boundary = true
        y .= τ .* Ad
        g .= g .+ y
        rNorm = knorm(n, g)

        if history
          qx += τ*alpha*gd + τ*alpha*kdot(n, d, y)/2
        end 
      
      else
        y .= alpha .* Ad
        g .= g .+ y
        rNorm = knorm(n, g)
        push!(H, s, y)

        if history
          qx += alpha*gd + alpha^2*dAd/2
        end 
        
      end

      if history
        push!(qxs, qx)
        push!(rNorms, rNorm)
      end

      # update stopping criteria
      user_requested_exit = callback(workspace) :: Bool
      
      resid_decrease_mach = (rNorm + one(T) ≤ one(T))
      resid_decrease_lim = rNorm ≤ ε
      resid_decrease = resid_decrease_lim || resid_decrease_mach
      solved = resid_decrease || on_boundary
      
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      
      overtimed = timer > timemax_ns
      
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm, start_time |> ktimer)
      
      # update inverse Hessian approximation
      if !(solved || tired || user_requested_exit || overtimed)
        
      end

    end

    # Termination status
    solved && (status == "unknown")   && (status = "solution good enough given atol and rtol")
    solved && on_boundary   && (status = "on trust-region boundary")
    tired                   && (status = "maximum number of iterations exceeded")
    user_requested_exit     && (status = "user-requested exit")
    overtimed               && (status = "time limit exceeded")

    # # Update x
    # warm_start && kaxpy!(n, one(FC), Δx, x)
    # workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
