@testset "lbfgs" begin
  lbfgs_tol = 1.0e-6
  for FC in (Float64,)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = lbfgs(A, b, itmax=10)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ lbfgs_tol)
      @test(stats.solved)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = lbfgs(A, b, radius=radius, itmax=10)
        @test(stats.solved)
        @test(abs(radius - norm(x)) ≤ lbfgs_tol * radius)
      end

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = lbfgs(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ lbfgs_tol)
      @test(stats.solved)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = lbfgs(A, b, radius=radius, itmax=10)
        @test(stats.solved)
        @test(abs(radius - norm(x)) ≤ lbfgs_tol * radius)
      end

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = lbfgs(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Test radius > 0  and b^TAb=0
      A, b = zero_rhs(FC=FC)
      solver = LbfgsWorkspace(A, b)
      lbfgs!(solver, A, b, radius = 10 * real(one(FC)))
      x, stats = solver.x, solver.stats
      @test stats.status == "x is a zero-residual solution"
      @test norm(x) == zero(FC)
      @test stats.niter == 0
      
      # Test residual of the solution with trust region
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      solver = LbfgsWorkspace(A, b)
      lbfgs!(solver, A, b; radius = 0.5 * real(one(FC)), history = true)
      x, stats, = solver.x, solver.stats
      r = b - A * x
      normr = norm(r)
      @test isapprox(normr, stats.residuals[end], atol=1.0e-8)
      @test stats.status == "nonpositive curvature detected"

      # test quadratic function values are computed correctly
      if FC == Float64
        A = FC[10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        solver = LbfgsWorkspace(A, b)
        lbfgs!(solver, A, b; radius = 10 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        qxs = stats.qvals
        q = -dot(b, x) + dot(x, A * x)/2
        @test length(qxs) == stats.niter + 1
        @test abs(qxs[end] - q) ≤ 1.0e-10
        @test abs(qxs[1]) ≤ 1.0e-10  # q(0) = 0
        # test that q is decreasing
        @test all(diff(qxs) .<= 1.0e-10)
      end

      # test quadratic function with trust-region
      if FC == Float64
        A = FC[
          10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 -1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        solver = LbfgsWorkspace(A, b)
        lbfgs!(solver, A, b; radius = 0.5 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        q = -dot(b, x) + dot(x, A * x)/2
        qxs = stats.qvals
        @test abs(q - qxs[end]) ≤ 1.0e-10
        @test stats.status == "nonpositive curvature detected"
      end

      # test quadratic function with warm start
      if FC == Float64  
        A = FC[
          10.0 0.0 0.0 0.0;
          0.0 8.0 0.0 0.0;
          0.0 0.0 5.0 0.0;
          0.0 0.0 0.0 -1.0
        ]
        b = FC[1.0, 1.0, 1.0, 0.1]
        x0 = FC[0.5, 0.5, 0.5, 0.05]
        solver = LbfgsWorkspace(A, b)
        lbfgs!(solver, A, b, x0; radius = 10 * real(one(FC)), history = true)
        x, stats, = solver.x, solver.stats
        q = -dot(b, x0) + dot(x0, A * x0)/2
        qxs = stats.qvals
        @test abs(q - qxs[1]) ≤ 1.0e-10
      end    

       # test callback function
      workspace = LbfgsWorkspace(A, b)
      tol = 1.0e-1
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      lbfgs!(workspace, A, b, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      @test_throws TypeError lbfgs(A, b, callback = workspace -> "string", history = true)
    end
  end
end