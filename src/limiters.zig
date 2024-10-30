const std = @import("std");
const mp = @import("multiproc.zig");
const fetch = @import("fetch.zig");

pub const LimiterTypes = enum {
    venkatakrishnan,
    barth_jeperson,
};

// Venkatakrishnan limiter main function
pub fn limiterVenkatakrishnan(
    phi: anytype,
    grad_phi: anytype,
    r_of: anytype,
    phi_minmax: anytype,
    k: anytype,
    volume: anytype,
) @TypeOf(phi) {
    const Type = @TypeOf(phi);
    const DType = @TypeOf(volume);
    const eps_sq = std.math.pow(DType, k, 3) * volume;

    if (@typeInfo(@TypeOf(phi)) == .Struct) {
        const del_2 = grad_phi.transpose().mulVec(r_of).scale(0.5);
        const del_1_min = phi_minmax[0].sub(phi);
        const del_1_max = phi_minmax[1].sub(phi);
        var limiter: Type = .{};

        inline for (0..limiter.data.len) |i| {
            limiter.data[i] = limiterVenkatakrishnanSubCalc(
                del_1_min.data[i],
                del_1_max.data[i],
                del_2.data[i],
                eps_sq,
            );
        }
        return limiter;
    } else {
        const del_2 = grad_phi.dot(r_of) * 0.5;
        const del_1_min = phi_minmax[0] - phi;
        const del_1_max = phi_minmax[1] - phi;

        const limiter = limiterVenkatakrishnanSubCalc(
            del_1_min,
            del_1_max,
            del_2,
            eps_sq,
        );
        return limiter;
    }
}

// Venkatakrishnan limiter sub function
pub fn limiterVenkatakrishnanSubCalc(
    del_1_min: anytype,
    del_1_max: anytype,
    del_2: anytype,
    eps_sq: anytype,
) @TypeOf(del_2) {
    const DType = @TypeOf(del_2);

    const del_2_sq = std.math.pow(DType, del_2, 2.0);
    var limiter: @TypeOf(del_2) = 1;

    if (del_2 > 0) {
        const del_1 = del_1_max;
        const del_1_sq = std.math.pow(DType, del_1, 2.0);
        const top = (del_1_sq + eps_sq) * del_2 + 2 * del_2_sq * del_1;
        const bot = del_1_sq + 2 * del_2_sq + del_1 * del_2 + eps_sq;
        limiter = top / bot / del_2;
    } else {
        if (del_2 < 0) {
            const del_1 = del_1_min;
            const del_1_sq = std.math.pow(DType, del_1, 2.0);
            const top = (del_1_sq + eps_sq) * del_2 + 2 * del_2_sq * del_1;
            const bot = del_1_sq + 2 * del_2_sq + del_1 * del_2 + eps_sq;
            limiter = top / bot / del_2;
        }
    }

    return limiter;
}

// Barth Jeperson limiter main function
pub fn limiterBarthJeperson(
    phi: anytype,
    grad_phi: anytype,
    r_of: anytype,
    phi_minmax: anytype,
) @TypeOf(phi) {
    const Type = @TypeOf(phi);

    if (@typeInfo(@TypeOf(phi)) == .Struct) {
        const del_2 = grad_phi.transpose().mulVec(r_of).scale(0.5);
        const del_1_min = phi_minmax[0].sub(phi);
        const del_1_max = phi_minmax[1].sub(phi);
        var limiter: Type = .{};
        inline for (0..limiter.data.len) |i| {
            limiter.data[i] = limiterBarthJepersonSubCalc(
                del_2.data[i],
                del_1_min.data[i],
                del_1_max.data[i],
            );
        }
        return limiter;
    } else {
        const del_2 = grad_phi.dot(r_of) * 0.5;
        const del_1_min = phi_minmax[0] - phi;
        const del_1_max = phi_minmax[1] - phi;
        const limiter = limiterBarthJepersonSubCalc(
            del_2,
            del_1_min,
            del_1_max,
        );
        return limiter;
    }
}

// Barth Jeperson limiter sub function
pub fn limiterBarthJepersonSubCalc(del_2: anytype, del_1_min: anytype, del_1_max: anytype) @TypeOf(del_2) {
    var limiter: @TypeOf(del_2) = 1.0;
    if (del_2 > 0.0) {
        limiter = @min(1.0, del_1_max / (@abs(del_2) + 1.0e-7));
    } else if (del_2 < 0.0) {
        limiter = @min(1.0, -del_1_min / (@abs(del_2) + 1.0e-7));
    } else {
        limiter = 1.0;
    }
    return limiter;
}

// Generic Limiter Function
pub fn unstructuredLimiter(
    comptime limiter_type: LimiterTypes,
    phi: anytype,
    grad_phi: anytype,
    r_of: anytype,
    phi_minmax: anytype,
    limiter_k: anytype,
    volume: anytype,
) @TypeOf(phi) {
    return switch (limiter_type) {
        LimiterTypes.barth_jeperson => limiterBarthJeperson(phi, grad_phi, r_of, phi_minmax),
        LimiterTypes.venkatakrishnan => limiterVenkatakrishnan(phi, grad_phi, r_of, phi_minmax, limiter_k, volume),
    };
}

//
pub fn kernelMinMaxVariables(
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

    const DType = @TypeOf(system.cells.items(.centroid)[0].data[0]);
    const VectorType = @TypeOf(system.cells.items(.v)[0]);

    for (thread_i_s..thread_i_f) |i| {
        // Get owner neighbour indices
        const i_o = system.faces.items(.i_owner)[i];
        const i_n = system.faces.items(.i_neighbour)[i];

        // Calculate minmax p
        const p_o = system.cells.items(.p)[i_o];
        const p_n = system.cells.items(.p)[i_n];

        _ = @atomicRmw(DType, &system.cells.items(.minmax_p)[i_o][0], .Min, p_n, .acq_rel);
        _ = @atomicRmw(DType, &system.cells.items(.minmax_p)[i_o][1], .Max, p_n, .acq_rel);

        _ = @atomicRmw(DType, &system.cells.items(.minmax_p)[i_n][0], .Min, p_o, .acq_rel);
        _ = @atomicRmw(DType, &system.cells.items(.minmax_p)[i_n][1], .Max, p_o, .acq_rel);

        // Calculate minmax v
        const v_o = system.cells.items(.v)[i_o];
        const v_n = system.cells.items(.v)[i_n];

        VectorType.atomicMinElem(&system.cells.items(.minmax_v)[i_o][0], v_n);
        VectorType.atomicMaxElem(&system.cells.items(.minmax_v)[i_o][1], v_n);

        VectorType.atomicMinElem(&system.cells.items(.minmax_v)[i_n][0], v_o);
        VectorType.atomicMaxElem(&system.cells.items(.minmax_v)[i_n][1], v_o);

        // Calculate minmax T
        const T_o = system.cells.items(.T)[i_o];
        const T_n = system.cells.items(.T)[i_n];

        _ = @atomicRmw(DType, &system.cells.items(.minmax_T)[i_o][0], .Min, T_n, .acq_rel);
        _ = @atomicRmw(DType, &system.cells.items(.minmax_T)[i_o][1], .Max, T_n, .acq_rel);

        _ = @atomicRmw(DType, &system.cells.items(.minmax_T)[i_n][0], .Min, T_o, .acq_rel);
        _ = @atomicRmw(DType, &system.cells.items(.minmax_T)[i_n][1], .Max, T_o, .acq_rel);
    }
}

pub fn computeMinMaxVariables(system: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelMinMaxVariables,
        .{},
        .{},
        system.n_faces,
        0,
    );
}

pub fn kernelUnstructuredLimiters(
    system: anytype,
    kernel_args: struct {},
    comptime comptime_kernel_args: struct {
        limiter_type: LimiterTypes,
    },
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = kernel_args;
    _ = thread_ind;

    const DType = @TypeOf(system.cells.items(.centroid)[0].data[0]);
    const VectorType = @TypeOf(system.cells.items(.v)[0]);

    // Compute Cell Limiters
    for (thread_i_s..thread_i_f) |i_f| {
        const i_o = system.faces.items(.i_owner)[i_f];
        const i_n = system.faces.items(.i_neighbour)[i_f];

        // Get face data
        const centroid_f = system.faces.items(.centroid)[i_f];

        // Get cell data owner
        const centroid_o = system.cells.items(.centroid)[i_o];
        const volume_o = system.cells.items(.volume)[i_o];
        const var_o = fetch.getVariables(system, i_o);
        const grad_o = fetch.getGradients(system, i_o);
        const minmax_o = fetch.getMinMax(system, i_o);

        // Get cell data neighbour
        const centroid_n = system.cells.items(.centroid)[i_n];
        const volume_n = system.cells.items(.volume)[i_n];
        const var_n = fetch.getVariables(system, i_n);
        const grad_n = fetch.getGradients(system, i_n);
        const minmax_n = fetch.getMinMax(system, i_n);

        // Relative positions
        const r_of = centroid_f.sub(centroid_o);
        const r_nf = centroid_f.sub(centroid_n);

        // Calculate limiter p
        const limiter_p_o_f = unstructuredLimiter(
            comptime_kernel_args.limiter_type,
            var_o.p,
            grad_o.grad_p,
            r_of,
            minmax_o.minmax_p,
            system.limiter_k,
            volume_o,
        );

        const limiter_p_n_f = unstructuredLimiter(
            comptime_kernel_args.limiter_type,
            var_n.p,
            grad_n.grad_p,
            r_nf,
            minmax_n.minmax_p,
            system.limiter_k,
            volume_n,
        );

        // Calculate limiter v
        const limiter_v_o_f = unstructuredLimiter(
            comptime_kernel_args.limiter_type,
            var_o.v,
            grad_o.grad_v,
            r_of,
            minmax_o.minmax_v,
            system.limiter_k,
            volume_o,
        );

        const limiter_v_n_f = unstructuredLimiter(
            comptime_kernel_args.limiter_type,
            var_n.v,
            grad_n.grad_v,
            r_nf,
            minmax_n.minmax_v,
            system.limiter_k,
            volume_n,
        );

        // Calculate limiter T
        const limiter_T_o_f = unstructuredLimiter(
            comptime_kernel_args.limiter_type,
            var_o.T,
            grad_o.grad_T,
            r_of,
            minmax_o.minmax_T,
            system.limiter_k,
            volume_o,
        );

        const limiter_T_n_f = unstructuredLimiter(
            comptime_kernel_args.limiter_type,
            var_n.T,
            grad_n.grad_T,
            r_nf,
            minmax_n.minmax_T,
            system.limiter_k,
            volume_n,
        );

        _ = @atomicRmw(DType, &system.cells.items(.limiter_p)[i_o], .Min, limiter_p_o_f, .acq_rel);
        _ = @atomicRmw(DType, &system.cells.items(.limiter_p)[i_n], .Min, limiter_p_n_f, .acq_rel);

        VectorType.atomicMinElem(&system.cells.items(.limiter_v)[i_o], limiter_v_o_f);
        VectorType.atomicMinElem(&system.cells.items(.limiter_v)[i_n], limiter_v_n_f);

        _ = @atomicRmw(DType, &system.cells.items(.limiter_T)[i_o], .Min, limiter_T_o_f, .acq_rel);
        _ = @atomicRmw(DType, &system.cells.items(.limiter_T)[i_n], .Min, limiter_T_n_f, .acq_rel);
    }
}

pub fn computeUnstructuredLimiters(
    system: anytype,
    comptime limiter_type: LimiterTypes,
) !void {
    try mp.computeMultiProc(
        system,
        kernelUnstructuredLimiters,
        .{},
        .{
            .limiter_type = limiter_type,
        },
        system.n_faces,
        0,
    );
}
