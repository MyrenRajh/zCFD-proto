const std = @import("std");
const mp = @import("multiproc.zig");

pub const TimeIntegrationSchemes = enum {
    exp_euler,
    exp_multi_rk_1_3,
    exp_multi_rk_1_4,
    exp_multi_rk_1_5,
    exp_multi_rk_2_3,
    exp_multi_rk_2_4,
    exp_multi_rk_2_5,
};

// Generic kernel for single explicit time step
pub fn kernelExplicitStep(
    system: anytype,
    kernel_args: struct {
        alpha: @TypeOf(system.cells.items(.p)[0]) = 1.0,
    },
    comptime comptime_kernel_args: struct {},
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) !void {
    _ = comptime_kernel_args;
    _ = thread_ind;

    const alpha = kernel_args.alpha;

    for (thread_i_s..thread_i_f) |i_o| {
        const volume = system.cells.items(.volume)[i_o];
        const dt = system.cells.items(.dt)[i_o];

        const p = system.cells.items(.p_old)[i_o];
        const v = system.cells.items(.v_old)[i_o];
        const T = system.cells.items(.T_old)[i_o];

        const rho = system.constitutive.density(p, T);
        const E = system.constitutive.energyTotal(T, v);

        const ddt_rho = system.cells.items(.ddt_rho)[i_o];
        const ddt_rho_v = system.cells.items(.ddt_rho_v)[i_o];
        const ddt_rho_E = system.cells.items(.ddt_rho_E)[i_o];

        const step_factor = alpha * dt / volume;

        const rho_new = rho + step_factor * ddt_rho;
        const rho_v_new = v.scale(rho).add(ddt_rho_v.scale(step_factor));
        const rho_E_new = (rho * E) + step_factor * ddt_rho_E;

        const v_new = rho_v_new.scale(1.0 / rho_new);
        const E_new = rho_E_new / rho_new;

        var T_new = system.constitutive.temperatureEnergyTotal(E_new, v_new);
        T_new = @max(T_new, 0);
        const p_new = system.constitutive.pressure(rho_new, T_new);

        system.cells.items(.p)[i_o] = p_new;
        system.cells.items(.v)[i_o] = v_new;
        system.cells.items(.T)[i_o] = T_new;

        system.cells.items(.minmax_p)[i_o] = .{ p_new, p_new };
        system.cells.items(.minmax_v)[i_o] = .{ v_new, v_new };
        system.cells.items(.minmax_T)[i_o] = .{ T_new, T_new };
    }
}

// Multithreaded kernel for single explicit time step
pub fn computeExplicitStep(system: anytype, alpha: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelExplicitStep,
        .{ .alpha = alpha },
        .{},
        system.n_cells,
        0,
    );
}

// Generic explicit time integration scheme
pub fn computeExplicitTimeIntegration(
    system: anytype,
    comptime scheme: TimeIntegrationSchemes,
) !void {
    const DType = @TypeOf(system.cells.items(.p)[0]);

    const stage_coeffs = switch (scheme) {
        .exp_euler => [1]DType{1.0},

        .exp_multi_rk_1_3 => [3]DType{ 0.1481, 0.4000, 1.0000 },
        .exp_multi_rk_1_4 => [4]DType{ 0.0833, 0.2069, 0.4265, 1.0000 },
        .exp_multi_rk_1_5 => [5]DType{ 0.0533, 0.1263, 0.2375, 0.4414, 1.0000 },

        .exp_multi_rk_2_3 => [3]DType{ 0.1918, 0.4929, 1.0000 },
        .exp_multi_rk_2_4 => [4]DType{ 0.1084, 0.2602, 0.5052, 1.0000 },
        .exp_multi_rk_2_5 => [5]DType{ 0.0695, 0.1602, 0.2898, 0.5060, 1.0000 },
    };

    try computeStoreOldSolution(system);
    for (stage_coeffs) |alpha| {
        try computeResetTemporalDerivative(system);
        try system.computeFluxes();
        try computeExplicitStep(system, alpha);
        try system.computePostStep();
    }
}

// Generic kernel for calculating local time step value
pub fn kernelLocalTimeStep(
    system: anytype,
    kernel_args: struct {
        thread_ind: usize = 0,
    },
    comptime comptime_kernel_args: struct {},
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = kernel_args;
    _ = comptime_kernel_args;
    _ = thread_ind;

    const VectorType = @TypeOf(system.cells.items(.v)[0]);

    for (thread_i_s..thread_i_f) |i_o| {
        const volume = system.cells.items(.volume)[i_o];

        const p = system.cells.items(.p)[i_o];
        const v = system.cells.items(.v)[i_o];
        const T = system.cells.items(.T)[i_o];

        const rho = system.constitutive.density(p, T);
        const mu = system.constitutive.dynamicViscosity(T);
        const Pr = system.constitutive.Pr;
        const gamma = system.constitutive.gamma;
        const c = system.constitutive.soundSpeed(T);

        const n_faces = system.cells.items(.n_faces)[i_o];
        const i_start_face = system.cells.items(.i_start_face)[i_o];
        var s_proj: VectorType = .{};

        for (i_start_face..i_start_face + n_faces) |i_cell_f| {
            const i_f = system.i_cell_faces.items[i_cell_f];
            const area = system.faces.items(.area)[i_f];
            const normal = system.faces.items(.normal)[i_f];
            const surface_vector = normal.scale(area);
            s_proj = s_proj.add(surface_vector.abs());
        }

        s_proj = s_proj.scale(0.5);

        const conv_spectral = v.abs().addScalar(c).mulElem(s_proj).sum();
        const visc_spectral = s_proj.mulElem(s_proj).scale(
            mu / Pr * @max(4.0 / (3.0 * rho), gamma / rho) / volume,
        ).sum();

        const local_dt_C_factor = system.local_dt_C_factor;
        const dt = volume / (conv_spectral + local_dt_C_factor * visc_spectral) * system.cfl;

        system.cells.items(.dt)[i_o] = dt;
    }
}

// Multithreaded kernel for calculating local time step value
pub fn computeLocalTimeStep(system: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelLocalTimeStep,
        .{},
        .{},
        system.n_cells,
        0,
    );
}

// Generic kernel for calculating global minimum timestep
pub fn kernelMinTimeStep(
    system: anytype,
    kernel_args: struct {
        p_thread_min_dt: *[]@TypeOf(system.cells.items(.p)[0]),
    },
    comptime comptime_kernel_args: struct {},
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = comptime_kernel_args;

    var min_dt = kernel_args.p_thread_min_dt.*[thread_ind];
    for (thread_i_s..thread_i_f) |i_o| {
        min_dt = @min(min_dt, system.cells.items(.dt)[i_o]);
    }

    kernel_args.p_thread_min_dt.*[thread_ind] = min_dt;
}

// Multithreaded kernel for calculating global minimum timestep
pub fn computeMinTimeStep(system: anytype) !@TypeOf(system.cells.items(.p)[0]) {
    const DType = @TypeOf(system.cells.items(.p)[0]);

    var thread_min_dt = try system.thread_allocator.alloc(DType, system.n_threads);
    defer system.thread_allocator.free(thread_min_dt);

    for (0..system.n_threads) |i| {
        thread_min_dt[i] = system.cells.items(.dt)[0];
    }

    try mp.computeMultiProc(
        system,
        kernelMinTimeStep,
        .{ .p_thread_min_dt = &thread_min_dt },
        .{},
        system.n_cells,
        0,
    );

    var global_min_dt = thread_min_dt[0];
    for (0..system.n_threads) |i| {
        global_min_dt = @min(global_min_dt, thread_min_dt[i]);
    }

    return global_min_dt;
}

// Generic kernel for asigning timestep
pub fn kernelAssignTimeStep(
    system: anytype,
    kernel_args: struct {
        dt: @TypeOf(system.cells.items(.p)[0]),
    },
    comptime comptime_kernel_args: struct {},
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = comptime_kernel_args;
    _ = thread_ind;

    const dt = kernel_args.dt;
    for (thread_i_s..thread_i_f) |i_o| {
        system.cells.items(.dt)[i_o] = dt;
    }
}

// Multithreaded kernel for calculating global minimum timestep
pub fn computeAssignTimeStep(system: anytype, dt: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelAssignTimeStep,
        .{ .dt = dt },
        .{},
        system.n_cells,
        0,
    );
}

// Generic kernel for storing old primitives
pub fn kernelStoreOldSolution(
    system: anytype,
    kernel_args: struct {},
    comptime comptime_kernel_args: struct {},
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = kernel_args;
    _ = comptime_kernel_args;
    _ = thread_ind;

    for (thread_i_s..thread_i_f) |i_o| {
        system.cells.items(.p_old)[i_o] = system.cells.items(.p)[i_o];
        system.cells.items(.v_old)[i_o] = system.cells.items(.v)[i_o];
        system.cells.items(.T_old)[i_o] = system.cells.items(.T)[i_o];
    }
}

// Multithreaded kernel for for storing old primitives
pub fn computeStoreOldSolution(system: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelStoreOldSolution,
        .{},
        .{},
        system.n_cells,
        0,
    );
}

// Generic kernel for storing old primitives
pub fn kernelResetTemporalDerivative(
    system: anytype,
    kernel_args: struct {},
    comptime comptime_kernel_args: struct {},
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = kernel_args;
    _ = comptime_kernel_args;
    _ = thread_ind;

    const VectorType = @TypeOf(system.cells.items(.v)[0]);

    for (thread_i_s..thread_i_f) |i_o| {
        system.cells.items(.ddt_rho)[i_o] = 0.0;
        system.cells.items(.ddt_rho_v)[i_o] = VectorType{};
        system.cells.items(.ddt_rho_E)[i_o] = 0.0;
    }
}

// Multithreaded kernel for for storing old primitives
pub fn computeResetTemporalDerivative(system: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelResetTemporalDerivative,
        .{},
        .{},
        system.n_cells,
        0,
    );
}
