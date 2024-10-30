const std = @import("std");
const flux = @import("flux.zig");
const recon = @import("reconstruction.zig");
const geom = @import("geometry.zig");
const mp = @import("multiproc.zig");
const fetch = @import("fetch.zig");

pub fn BoundaryData(DType: anytype, VectorType: anytype, MatrixType: anytype) type {
    return struct {
        p_b: DType,
        v_b: VectorType,
        T_b: DType,
        grad_p_b: VectorType,
        grad_v_b: MatrixType,
        grad_T_b: VectorType,
    };
}

pub const BCType = enum {
    symmetry, //
    wall_viscous_heated, //
    wall_inviscid_heated,
    wall_viscous_isothermal, //
    wall_inviscid_isothermal,
    inlet_supersonic, //
    inlet_subsonic_velocity,
    inlet_subsonic_static_pressure,
    inlet_subsonic_total_pressure,
    outlet_supersonic, //
    outlet_subsonic_pressure,
    outlet_subsonic_mass_flow,
    outlet_subsonic_developed,
    farfield_subsonic_inflow,
    farfield_subsonic_outflow,
    farfield_supersonic_inflow,
    farfield_supersonic_outflow,
};

pub fn BoundaryInfo(DType: type, VectorType: type) type {
    return struct {
        n_faces: usize = 0,
        i_start_face: usize = 0,
        type: BCType = .symmetry,
        p: DType = 0,
        v: VectorType = .{},
        T: DType = 0,
        grad_T_normal: DType = 0,
        grad_v_normal: VectorType = .{},
        grad_p_normal: DType = 0,
    };
}

pub fn computeBCWallViscousHeated(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const MatrixType = @TypeOf(grad_o.grad_v);

    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_b = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_of);
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity condition
    const v_b = boundary_info.v;
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_b_local = local_basis.transform_mat_T.mulVec(v_b);
    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);

    var ddn_v_local = v_b_local.sub(v_oo_n_local).scale(1.0 / d_of_n);
    ddn_v_local.data[0] = 0;

    var grad_v_b_local: MatrixType = .{};
    grad_v_b_local.data[0] = ddn_v_local.data;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Condition
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = boundary_info.grad_T_normal;

    const T_b = boundary_info.grad_T_normal * d_of_n + T_oo_n;
    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCWallViscousIsothermal(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const MatrixType = @TypeOf(grad_o.grad_v);

    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_b = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_of);
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity condition
    const v_b = boundary_info.v;
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_b_local = local_basis.transform_mat_T.mulVec(v_b);
    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);

    var ddn_v_local = v_b_local.sub(v_oo_n_local).scale(1.0 / d_of_n);
    ddn_v_local.data[0] = 0;

    var grad_v_b_local: MatrixType = .{};
    grad_v_b_local.data[0] = ddn_v_local.data;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Condition
    const T_b = boundary_info.T;
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = (T_b - T_oo_n) / d_of_n;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCWallInviscidHeated(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const MatrixType = @TypeOf(grad_o.grad_v);

    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_b = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_of);
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity condition
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);
    var v_b_local = v_oo_n_local;
    v_b_local.data[0] = 0.0;

    const grad_v_b: MatrixType = .{};
    const v_b = local_basis.transform_mat.mulVec(v_b_local);

    // Temperature Condition
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = boundary_info.grad_T_normal;

    const T_b = boundary_info.grad_T_normal * d_of_n + T_oo_n;
    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCWallInviscidIsothermal(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const MatrixType = @TypeOf(grad_o.grad_v);

    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_b = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_of);
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity condition
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);
    var v_b_local = v_oo_n_local;
    v_b_local.data[0] = 0.0;

    const grad_v_b: MatrixType = .{};
    const v_b = local_basis.transform_mat.mulVec(v_b_local);

    // Temperature Condition
    const T_b = boundary_info.T;
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = (T_b - T_oo_n) / d_of_n;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCInletSupersonic(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_b = boundary_info.p;
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity condition
    const v_b = boundary_info.v;
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_b_local = local_basis.transform_mat_T.mulVec(v_b);
    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);

    const ddn_v_local = v_b_local.sub(v_oo_n_local).scale(1.0 / d_of_n);

    var grad_v_b_local = local_basis.transform_mat_T.mul(grad_v_o).mul(local_basis.transform_mat);
    grad_v_b_local.data[0] = ddn_v_local.data;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Condition
    const T_b = boundary_info.T;
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = (T_b - T_oo_n) / d_of_n;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCOutletSupersonic(
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    // Types
    const VectorType = @TypeOf(var_o.v);

    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Pressure Condition
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);
    const p_b = p_oo_n;

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = 0.0;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);
    // const grad_p_b: VectorType = .{};

    // Velocity condition
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);
    const v_b = v_oo_n;

    const ddn_v_local: VectorType = .{};

    var grad_v_b_local = local_basis.transform_mat_T.mul(grad_v_o).mul(local_basis.transform_mat);
    grad_v_b_local.data[0] = ddn_v_local.data;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Condition
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);
    const T_b = T_oo_n;

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = 0.0;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCInletSubsonicVelocity(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
    constitutive: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Properties
    const c_p = constitutive.c_p;
    const gamma = constitutive.gamma;
    const gamma_n1 = gamma - 1.0;

    // Temperature Interior
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    // Velocity Interior
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    // Riemann Invariant
    const c_oo_n = constitutive.soundSpeed(T_oo_n);
    const riemann_inv = v_oo_n.dot(local_geom.normal) - 2.0 * c_oo_n / gamma_n1;

    const cos_theta = -v_oo_n.dot(local_geom.normal) / v_oo_n.mag();
    const c0_sq = c_oo_n * c_oo_n + gamma_n1 / 2.0 * v_oo_n.dot(v_oo_n);

    const c_b_factor = -riemann_inv / (cos_theta * cos_theta + 2.0);
    const c_b_bracket = 1.0 + cos_theta * @sqrt(
        (gamma_n1 * cos_theta * cos_theta + 2.0) * c0_sq / (gamma_n1 * riemann_inv * riemann_inv) - gamma_n1 / 2.0,
    );

    const c_b = c_b_factor * c_b_bracket;
    const c_ratio = c_b * c_b / c0_sq;

    const v_inlet = boundary_info.v;
    const T_inlet = boundary_info.T;
    const p_inlet = boundary_info.T;

    const c_inlet = constitutive.soundSpeed(T_inlet);
    const mach_inlet = constitutive.machNumber(c_inlet, v_inlet);

    const T_0 = T_inlet * (1.0 + gamma_n1 / 2.0 * mach_inlet * mach_inlet);
    const p_0 = p_inlet * std.math.pow(
        @TypeOf(T_0),
        1.0 + gamma_n1 / 2.0 * mach_inlet * mach_inlet,
        gamma / gamma_n1,
    );

    const T_b = T_0 * c_ratio;
    const p_b = p_0 * std.math.pow(
        @TypeOf(T_0),
        T_b / T_0,
        gamma / gamma_n1,
    );

    const v_b_mag = @sqrt(2.0 * c_p * (T_0 - T_b));
    const v_b = v_inlet.unit().scale(v_b_mag);

    // Pressure Gradient
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity Gradient
    const v_b_local = local_basis.transform_mat_T.mulVec(v_b);
    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);

    const ddn_v_local = v_b_local.sub(v_oo_n_local).scale(1.0 / d_of_n);

    var grad_v_b_local = local_basis.transform_mat_T.mul(grad_v_o).mul(local_basis.transform_mat);
    grad_v_b_local.data[0] = ddn_v_local.data;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Gradient
    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = (T_b - T_oo_n) / d_of_n;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCOutletSubsonicPressure(
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
    constitutive: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_b = boundary_info.p;
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);

    //Temperature Condition
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);

    const rho_oo_n = constitutive.density(p_oo_n, T_oo_n);
    const c_oo_n = constitutive.soundSpeed(T_oo_n);

    const rho_b = rho_oo_n + (p_b - p_oo_n) / (c_oo_n * c_oo_n);
    const T_b = constitutive.temperature(p_b, rho_b);

    // Velocity condition
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_b = c_oo_n + local_basis.normal.scale(
        (p_oo_n - p_b) / (rho_oo_n * c_oo_n),
    );

    // Pressure Gradient
    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = (p_b - p_oo_n) / d_of_n;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity Gradient
    const v_b_local = local_basis.transform_mat_T.mulVec(v_b);
    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);

    const ddn_v_local = v_b_local.sub(v_oo_n_local).scale(1.0 / d_of_n);

    var grad_v_b_local = local_basis.transform_mat_T.mul(grad_v_o).mul(local_basis.transform_mat);
    grad_v_b_local.data[0] = ddn_v_local.data;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Gradient
    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = (T_b - T_oo_n) / d_of_n;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn computeBCSymmetry(
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    const p_o = var_o.p;
    const v_o = var_o.v;
    const T_o = var_o.T;
    const grad_p_o = grad_o.grad_p;
    const grad_v_o = grad_o.grad_v;
    const grad_T_o = grad_o.grad_T;

    // Geometry
    const d_of_n = local_geom.r_of_n.mag();

    // Pressure Condition
    const p_oo_n = recon.exPWLinScalar(p_o, grad_p_o, local_geom.r_oo_n);
    const p_b = p_oo_n;

    var grad_p_b_local = local_basis.transform_mat_T.mulVec(grad_p_o);
    grad_p_b_local.data[0] = 0.0;

    const grad_p_b = local_basis.transform_mat.mulVec(grad_p_b_local);

    // Velocity condition
    const v_oo_n = recon.exPWLinVector(v_o, grad_v_o, local_geom.r_oo_n);

    const v_oo_n_local = local_basis.transform_mat_T.mulVec(v_oo_n);
    var v_b_local = v_oo_n_local;
    v_b_local.data[0] = 0.0;

    const v_b = local_basis.transform_mat.mulVec(v_b_local);

    var grad_v_b_local = local_basis.transform_mat_T.mul(grad_v_o).mul(local_basis.transform_mat);
    grad_v_b_local.data[0][0] = -v_oo_n_local.data[0] / d_of_n;
    grad_v_b_local.data[0][1] = 0.0;
    grad_v_b_local.data[0][2] = 0.0;
    grad_v_b_local.data[1][0] = 0.0;
    grad_v_b_local.data[2][0] = 0.0;

    const grad_v_b = local_basis.transform_mat.mul(grad_v_b_local).mul(local_basis.transform_mat_T);

    // Temperature Condition
    const T_oo_n = recon.exPWLinScalar(T_o, grad_T_o, local_geom.r_oo_n);
    const T_b = T_oo_n;

    var grad_T_b_local = local_basis.transform_mat_T.mulVec(grad_T_o);
    grad_T_b_local.data[0] = 0.0;

    const grad_T_b = local_basis.transform_mat.mulVec(grad_T_b_local);

    return .{ .p_b = p_b, .v_b = v_b, .T_b = T_b, .grad_p_b = grad_p_b, .grad_v_b = grad_v_b, .grad_T_b = grad_T_b };
}

pub fn calcBoundaryData(
    comptime bc_type: BCType,
    boundary_info: anytype,
    local_geom: anytype,
    local_basis: anytype,
    var_o: anytype,
    grad_o: anytype,
    constitutive: anytype,
) BoundaryData(@TypeOf(var_o.p), @TypeOf(var_o.v), @TypeOf(grad_o.grad_v)) {
    return switch (bc_type) {
        BCType.symmetry => computeBCSymmetry(
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.wall_viscous_heated => computeBCWallViscousHeated(
            boundary_info,
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.wall_viscous_isothermal => computeBCWallViscousIsothermal(
            boundary_info,
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.wall_inviscid_heated => computeBCWallInviscidHeated(
            boundary_info,
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.wall_inviscid_isothermal => computeBCWallInviscidIsothermal(
            boundary_info,
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.inlet_supersonic => computeBCInletSupersonic(
            boundary_info,
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.outlet_supersonic => computeBCOutletSupersonic(
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
        BCType.inlet_subsonic_velocity => computeBCInletSubsonicVelocity(
            local_geom,
            local_basis,
            var_o,
            grad_o,
            constitutive,
        ),
        BCType.outlet_subsonic_pressure => computeBCOutletSubsonicPressure(
            local_geom,
            local_basis,
            var_o,
            grad_o,
            constitutive,
        ),
        else => computeBCSymmetry(
            local_geom,
            local_basis,
            var_o,
            grad_o,
        ),
    };
}

pub fn kernelBoundaryCellData(
    system: anytype,
    kernel_args: struct {
        boundary_info: *const BoundaryInfo(
            @TypeOf(system.cells.items(.p)[0]),
            @TypeOf(system.cells.items(.v)[0]),
        ),
    },
    comptime comptime_kernel_args: struct {
        bc_type: BCType,
    },
    thread_i_s: usize,
    thread_i_f: usize,
    thread_ind: usize,
) !void {
    _ = thread_ind;
    for (thread_i_s..thread_i_f) |i_f| {
        const i_o = system.faces.items(.i_owner)[i_f];
        const i_n = system.faces.items(.i_neighbour)[i_f];

        // Face geometric info
        const centroid_f = system.faces.items(.centroid)[i_f];
        const normal = system.faces.items(.normal)[i_f];
        const tangent = system.faces.items(.tangent)[i_f];

        // Owner cell geometric info
        const centroid_o = system.cells.items(.centroid)[i_o];

        // Owner cell flow info
        const var_o = fetch.getVariables(system, i_o);
        const grad_o = fetch.getGradients(system, i_o);

        // Calculate Geomtetric info
        const local_basis = geom.localNormalBasis(normal, tangent);
        const local_geom = geom.localGeometry(centroid_o, centroid_f, local_basis);

        // Calculate Boundary Values
        const bc_vals = calcBoundaryData(
            comptime_kernel_args.bc_type,
            kernel_args.boundary_info,
            local_geom,
            local_basis,
            var_o,
            grad_o,
            &system.constitutive,
        );

        // Save calculated wall values in ghost cell
        system.cells.items(.p)[i_n] = bc_vals.p_b;
        system.cells.items(.v)[i_n] = bc_vals.v_b;
        system.cells.items(.T)[i_n] = bc_vals.T_b;

        system.cells.items(.grad_p)[i_n] = bc_vals.grad_p_b;
        system.cells.items(.grad_v)[i_n] = bc_vals.grad_v_b;
        system.cells.items(.grad_T)[i_n] = bc_vals.grad_T_b;
    }
}

pub fn computeBoundaryCellData(
    system: anytype,
    boundary_info: anytype,
    comptime bc_type: BCType,
) !void {
    try mp.computeMultiProc(
        system,
        kernelBoundaryCellData,
        .{ .boundary_info = boundary_info },
        .{ .bc_type = bc_type },
        boundary_info.n_faces,
        boundary_info.i_start_face,
    );
}

pub fn computeBCs(system: anytype) !void {
    for (0..system.n_boundary_patches) |i| {
        const boundary_info = system.boundaries.get(i);

        switch (boundary_info.type) {
            BCType.symmetry => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.symmetry,
            ),
            BCType.wall_viscous_heated => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.wall_viscous_heated,
            ),
            BCType.wall_viscous_isothermal => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.wall_viscous_isothermal,
            ),
            BCType.wall_inviscid_heated => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.wall_inviscid_heated,
            ),
            BCType.wall_inviscid_isothermal => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.wall_inviscid_isothermal,
            ),
            BCType.inlet_supersonic => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.inlet_supersonic,
            ),
            BCType.outlet_supersonic => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.outlet_supersonic,
            ),
            else => try computeBoundaryCellData(
                system,
                &boundary_info,
                BCType.symmetry,
            ),
        }
    }
}
