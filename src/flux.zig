const std = @import("std");
const recon = @import("reconstruction.zig");
const riemann = @import("riemann.zig");
const fetch = @import("fetch.zig");
const mp = @import("multiproc.zig");

pub const ConvectiveFluxScheme = enum {
    none,
    riemann_roe,
    riemann_hllc,
};

pub const ViscousFluxScheme = enum {
    none,
    interp_grad_linear,
};

// Generic flux struct
pub fn Flux(DataType: anytype, VectorType: anytype) type {
    return struct {
        flux_rho: DataType = 0.0,
        flux_rho_v: VectorType = .{},
        flux_rho_E: DataType = 0.0,
    };
}

// Compute viscous flux from interpolated primitives and their gradients
pub fn viscousFlux(
    v: anytype,
    tau: anytype,
    grad_T: anytype,
    k: anytype,
    normal: anytype,
) Flux(@TypeOf(k), @TypeOf(v)) {
    const flux_rho_v = tau.mulVec(normal);
    const flux_rho_E = tau.mulVec(v).add(grad_T.scale(k)).dot(normal);
    return .{ .flux_rho = 0.0, .flux_rho_v = flux_rho_v, .flux_rho_E = flux_rho_E };
}

// Compute convective flux from primitives
pub fn convectiveFlux(
    p: anytype,
    v: anytype,
    rho: anytype,
    H: anytype,
    normal: anytype,
) Flux(@TypeOf(p), @TypeOf(v)) {
    const v_contra = v.dot(normal);
    const flux_rho = rho * v_contra;
    const flux_rho_v = v.scale(flux_rho).add(normal.scale(p));
    const flux_rho_E = rho * H * v_contra;
    return .{ .flux_rho = flux_rho, .flux_rho_v = flux_rho_v, .flux_rho_E = flux_rho_E };
}

// Compute neighbour extrapolation
pub fn faceConvectiveFlux(
    comptime scheme: ConvectiveFluxScheme,
    p_LR: anytype,
    v_LR: anytype,
    rho_LR: anytype,
    H_LR: anytype,
    normal: anytype,
    eos: anytype,
) Flux(@TypeOf(p_LR.L), @TypeOf(v_LR.L)) {
    return switch (scheme) {
        .riemann_roe => riemann.nsRoe(p_LR, v_LR, rho_LR, H_LR, normal, eos),
        .riemann_hllc => riemann.nsHLLC(p_LR, v_LR, rho_LR, H_LR, normal, eos),
        else => .{},
    };
}

pub fn faceViscousFlux(
    comptime scheme: ViscousFluxScheme,
    v_f: anytype,
    tau_f: anytype,
    grad_T_f: anytype,
    k_f: anytype,
    normal: anytype,
) Flux(@TypeOf(k_f), @TypeOf(v_f)) {
    return switch (scheme) {
        .interp_grad_linear => viscousFlux(v_f, tau_f, grad_T_f, k_f, normal),
        else => .{},
    };
}

// Generic face flux kernel
pub fn kernelInternalFaceFluxes(
    system: anytype,
    kernel_args: struct {},
    comptime comptime_kernel_args: struct {
        conv_flux_scheme: ConvectiveFluxScheme,
        visc_flux_scheme: ViscousFluxScheme,
    },
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = kernel_args;
    _ = thread_ind;

    const DataType = @TypeOf(system.cells.items(.p)[0]);
    const VectorType = @TypeOf(system.cells.items(.v)[0]);

    for (thread_i_s..thread_i_f) |i_f| {
        // Get owner neighbour indices
        const i_o = system.faces.items(.i_owner)[i_f];
        const i_n = system.faces.items(.i_neighbour)[i_f];

        // Get face data
        const centroid_f = system.faces.items(.centroid)[i_f];
        const area = system.faces.items(.area)[i_f];
        const normal = system.faces.items(.normal)[i_f];

        // Get cell data owner
        const centroid_o = system.cells.items(.centroid)[i_o];
        const var_o = fetch.getVariables(system, i_o);
        const grad_o = fetch.getGradients(system, i_o);
        const lim_o = fetch.getLimiters(system, i_o);

        // Get cell data neighbour
        const centroid_n = system.cells.items(.centroid)[i_n];
        const var_n = fetch.getVariables(system, i_n);
        const grad_n = fetch.getGradients(system, i_n);
        const lim_n = fetch.getLimiters(system, i_n);

        // Relative positions
        const r_on = centroid_n.sub(centroid_o);
        const r_of = centroid_f.sub(centroid_o);
        const r_nf = centroid_f.sub(centroid_n);

        // Compute LR states
        const p_L = recon.exPWLinLimScalar(var_o.p, grad_o.grad_p, r_of, lim_o.limiter_p);
        const v_L = recon.exPWLinLimVector(var_o.v, grad_o.grad_v, r_of, lim_o.limiter_v);
        const T_L = recon.exPWLinLimScalar(var_o.T, grad_o.grad_T, r_of, lim_o.limiter_T);

        const p_R = recon.exPWLinLimScalar(var_n.p, grad_n.grad_p, r_nf, lim_n.limiter_p);
        const v_R = recon.exPWLinLimVector(var_n.v, grad_n.grad_v, r_nf, lim_n.limiter_v);
        const T_R = recon.exPWLinLimScalar(var_n.T, grad_n.grad_T, r_nf, lim_n.limiter_T);

        const rho_L = system.constitutive.density(p_L, T_L);
        const rho_R = system.constitutive.density(p_R, T_R);

        const H_L = system.constitutive.enthalpyTotal(T_L, v_L);
        const H_R = system.constitutive.enthalpyTotal(T_R, v_R);

        const p_LR = .{ .L = p_L, .R = p_R };
        const v_LR = .{ .L = v_L, .R = v_R };
        const rho_LR = .{ .L = rho_L, .R = rho_R };
        const H_LR = .{ .L = H_L, .R = H_R };

        // Interpolate face variables
        const v_f = recon.inPWLinVector(var_o.v, var_n.v, grad_o.grad_v, grad_n.grad_v, r_of, r_nf);
        const T_f = recon.inPWLinScalar(var_o.T, var_n.T, grad_o.grad_T, grad_n.grad_T, r_of, r_nf);

        const grad_v_f = recon.inLinGradVector(var_o.v, var_n.v, grad_o.grad_v, grad_n.grad_v, r_of, r_nf, r_on);
        const grad_T_f = recon.inLinGradScalar(var_o.T, var_n.T, grad_o.grad_T, grad_n.grad_T, r_of, r_nf, r_on);

        const mu_f = system.constitutive.dynamicViscosity(T_f);
        const k_f = system.constitutive.thermalConductivity(mu_f);
        const tau_f = system.constitutive.viscousStress(mu_f, grad_v_f);

        // Compute Fluxes
        const conv_flux = faceConvectiveFlux(comptime_kernel_args.conv_flux_scheme, p_LR, v_LR, rho_LR, H_LR, normal, &system.constitutive);

        const visc_flux = faceViscousFlux(comptime_kernel_args.visc_flux_scheme, v_f, tau_f, grad_T_f, k_f, normal);

        // Sum Fluxes
        const flux_rho = area * (visc_flux.flux_rho - conv_flux.flux_rho);
        const flux_rho_v = visc_flux.flux_rho_v.sub(conv_flux.flux_rho_v).scale(area);
        const flux_rho_E = area * (visc_flux.flux_rho_E - conv_flux.flux_rho_E);

        // Add flux to owner cells
        _ = @atomicRmw(DataType, &system.cells.items(.ddt_rho)[i_o], .Add, flux_rho, .acq_rel);
        VectorType.atomicAdd(&system.cells.items(.ddt_rho_v)[i_o], flux_rho_v);
        _ = @atomicRmw(DataType, &system.cells.items(.ddt_rho_E)[i_o], .Add, flux_rho_E, .acq_rel);

        // Add flux to neighbour cells
        _ = @atomicRmw(DataType, &system.cells.items(.ddt_rho)[i_n], .Sub, flux_rho, .acq_rel);
        VectorType.atomicSub(&system.cells.items(.ddt_rho_v)[i_n], flux_rho_v);
        _ = @atomicRmw(DataType, &system.cells.items(.ddt_rho_E)[i_n], .Sub, flux_rho_E, .acq_rel);
    }
}

pub fn kernelBoundaryFaceFluxes(
    system: anytype,
    kernel_args: struct {},
    comptime comptime_kernel_args: struct {
        conv_flux_scheme: ConvectiveFluxScheme,
        visc_flux_scheme: ViscousFluxScheme,
    },
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) void {
    _ = kernel_args;
    _ = thread_ind;

    const DataType = @TypeOf(system.cells.items(.p)[0]);
    const VectorType = @TypeOf(system.cells.items(.v)[0]);

    for (thread_i_s..thread_i_f) |i_f| {
        // Get owner neighbour indices
        const i_o = system.faces.items(.i_owner)[i_f];
        const i_n = system.faces.items(.i_neighbour)[i_f];

        // Get face data
        const centroid_f = system.faces.items(.centroid)[i_f];
        const area = system.faces.items(.area)[i_f];
        const normal = system.faces.items(.normal)[i_f];

        // Get cell data owner
        const centroid_o = system.cells.items(.centroid)[i_o];
        const var_o = fetch.getVariables(system, i_o);
        const grad_o = fetch.getGradients(system, i_o);
        const lim_o = fetch.getLimiters(system, i_o);

        // Get cell data neighbour
        const var_n = fetch.getVariables(system, i_n);
        const grad_n = fetch.getGradients(system, i_n);

        // Relative positions
        const r_of = centroid_f.sub(centroid_o);

        // Compute LR states
        const p_L = recon.exPWLinLimScalar(var_o.p, grad_o.grad_p, r_of, lim_o.limiter_p);
        const v_L = recon.exPWLinLimVector(var_o.v, grad_o.grad_v, r_of, lim_o.limiter_v);
        const T_L = recon.exPWLinLimScalar(var_o.T, grad_o.grad_T, r_of, lim_o.limiter_T);

        const p_R: DataType = var_n.p;
        const v_R: VectorType = var_n.v;
        const T_R: DataType = var_n.T;

        const rho_L = system.constitutive.density(p_L, T_L);
        const rho_R = system.constitutive.density(p_R, T_R);

        const H_L = system.constitutive.enthalpyTotal(T_L, v_L);
        const H_R = system.constitutive.enthalpyTotal(T_R, v_R);

        const p_LR = .{ .L = p_L, .R = p_R };
        const v_LR = .{ .L = v_L, .R = v_R };
        const rho_LR = .{ .L = rho_L, .R = rho_R };
        const H_LR = .{ .L = H_L, .R = H_R };

        const mu_f = system.constitutive.dynamicViscosity(T_R);
        const k_f = system.constitutive.thermalConductivity(mu_f);
        const tau_f = system.constitutive.viscousStress(mu_f, grad_n.grad_v);

        // Compute Fluxes
        const conv_flux = faceConvectiveFlux(
            comptime_kernel_args.conv_flux_scheme,
            p_LR,
            v_LR,
            rho_LR,
            H_LR,
            normal,
            &system.constitutive,
        );

        const visc_flux = faceViscousFlux(
            comptime_kernel_args.visc_flux_scheme,
            var_n.v,
            tau_f,
            grad_n.grad_T,
            k_f,
            normal,
        );

        // Sum Fluxes
        const flux_rho = area * (visc_flux.flux_rho - conv_flux.flux_rho);
        const flux_rho_v = visc_flux.flux_rho_v.sub(conv_flux.flux_rho_v).scale(area);
        const flux_rho_E = area * (visc_flux.flux_rho_E - conv_flux.flux_rho_E);

        // Add flux to owner cells
        _ = @atomicRmw(DataType, &system.cells.items(.ddt_rho)[i_o], .Add, flux_rho, .acq_rel);
        VectorType.atomicAdd(&system.cells.items(.ddt_rho_v)[i_o], flux_rho_v);
        _ = @atomicRmw(DataType, &system.cells.items(.ddt_rho_E)[i_o], .Add, flux_rho_E, .acq_rel);
    }
}

// Multithreaded kernel for face fluxes
pub fn computeFaceFluxes(
    system: anytype,
    comptime conv_flux_scheme: ConvectiveFluxScheme,
    comptime visc_flux_scheme: ViscousFluxScheme,
) !void {
    try mp.computeMultiProc(
        system,
        kernelInternalFaceFluxes,
        .{},
        .{
            .conv_flux_scheme = conv_flux_scheme,
            .visc_flux_scheme = visc_flux_scheme,
        },
        system.n_faces_internal,
        0,
    );

    try mp.computeMultiProc(
        system,
        kernelBoundaryFaceFluxes,
        .{},
        .{
            .conv_flux_scheme = conv_flux_scheme,
            .visc_flux_scheme = visc_flux_scheme,
        },
        system.n_faces_external,
        system.n_faces_internal,
    );
}
