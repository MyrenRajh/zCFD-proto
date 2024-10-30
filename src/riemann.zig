const std = @import("std");
const flux = @import("flux.zig");

// Compute roe averages
pub fn roeAverage(
    p_LR: anytype,
    v_LR: anytype,
    rho_LR: anytype,
    H_LR: anytype,
    normal: anytype,
    eos: anytype,
) struct {
    rho: @TypeOf(p_LR.L),
    v: @TypeOf(v_LR.L),
    H: @TypeOf(p_LR.L),
    c: @TypeOf(p_LR.L),
    v_contra: @TypeOf(p_LR.L),
    q_sq: @TypeOf(p_LR.L),
} {
    // Pre-compute sqrt density
    const sqrt_rho_L = @sqrt(rho_LR.L);
    const sqrt_rho_R = @sqrt(rho_LR.R);
    const inv_sqrt_rho_sum = 1.0 / (sqrt_rho_L + sqrt_rho_R);

    // Compute roe averages
    const rho = @sqrt(rho_LR.L * rho_LR.R);
    const v = v_LR.L.scale(sqrt_rho_L).add(v_LR.R.scale(sqrt_rho_R)).scale(inv_sqrt_rho_sum);
    const H = (H_LR.L * sqrt_rho_L + H_LR.R * sqrt_rho_R) * inv_sqrt_rho_sum;
    const q_sq = v.dot(v);
    const v_contra = v.dot(normal);
    const c = eos.soundSpeedEnthalpy(H, v);

    return .{ .rho = rho, .v = v, .H = H, .c = c, .v_contra = v_contra, .q_sq = q_sq };
}

pub fn nsRoe(
    p_LR: anytype,
    v_LR: anytype,
    rho_LR: anytype,
    H_LR: anytype,
    normal: anytype,
    eos: anytype,
) flux.Flux(@TypeOf(p_LR.L), @TypeOf(v_LR.L)) {
    // Compute left right fluxes
    const flux_L = flux.convectiveFlux(p_LR.L, v_LR.L, rho_LR.L, H_LR.L, normal);
    const flux_R = flux.convectiveFlux(p_LR.R, v_LR.R, rho_LR.R, H_LR.R, normal);

    // Compute roe averages
    const roe_ave = roeAverage(p_LR, v_LR, rho_LR, H_LR, normal, eos);
    const rho = roe_ave.rho;
    const v = roe_ave.v;
    const H = roe_ave.H;
    const c = roe_ave.c;
    const v_contra = roe_ave.v_contra;
    const q_sq = roe_ave.q_sq;

    // Pre-compute jump conditions
    const del_rho = rho_LR.L - rho_LR.R;
    const del_p = p_LR.L - p_LR.R;
    const del_v = v_LR.L.sub(v_LR.R);
    const del_v_contra = del_v.dot(normal);

    // Compute roe matrix corrected eigenvalues
    const eig_1 = entropyCorrectionHarten(@abs(v_contra - c), c);
    const eig_2 = entropyCorrectionHarten(@abs(v_contra), c);
    const eig_3 = entropyCorrectionHarten(@abs(v_contra + c), c);

    // Pre-compute coefficients for efficiency
    const coeff_1 = eig_1 * (del_p - rho * c * del_v_contra) / (2 * c * c);
    const coeff_2_1 = eig_2 * (del_rho - del_p / (2 * c * c));
    const coeff_2_2 = eig_2 * rho;
    const coeff_3 = eig_3 * (del_p + rho * c * del_v_contra) / (2 * c * c);

    // Compute roe flux difference continuity
    const flux_diff_rho = coeff_1 + coeff_2_1 + coeff_3;

    // Compute roe flux difference momentum
    const flux_diff_rho_v = v.sub(normal.scale(c)).scale(coeff_1).add(
        v.scale(coeff_2_1).add(del_v.sub(normal.scale(del_v_contra)).scale(coeff_2_2)),
    ).add(
        v.add(normal.scale(c)).scale(coeff_3),
    );

    // Compute roe flux difference energy
    const flux_diff_rho_E = coeff_1 * (H - c * v_contra) +
        coeff_2_1 * 0.5 * q_sq + coeff_2_2 * (v.dot(del_v) - v_contra * del_v_contra) +
        coeff_3 * (H + c * v_contra);

    // Compute Final Fluxes
    const flux_rho = (flux_L.flux_rho + flux_R.flux_rho - flux_diff_rho) * 0.5;
    const flux_rho_v = flux_L.flux_rho_v.add(flux_R.flux_rho_v).sub(flux_diff_rho_v).scale(0.5);
    const flux_rho_E = (flux_L.flux_rho_E + flux_R.flux_rho_E - flux_diff_rho_E) * 0.5;

    return .{ .flux_rho = flux_rho, .flux_rho_v = flux_rho_v, .flux_rho_E = flux_rho_E };
}

pub fn entropyCorrectionHarten(
    abs_eig: anytype,
    c: anytype,
) @TypeOf(abs_eig) {
    const delta = c / 20.0;
    var correct_eig = abs_eig;

    if (abs_eig <= delta) {
        correct_eig = (abs_eig * abs_eig + delta * delta) / (2 * delta);
    }

    return correct_eig;
}

pub fn nsHLLC(
    p_LR: anytype,
    v_LR: anytype,
    rho_LR: anytype,
    H_LR: anytype,
    normal: anytype,
    eos: anytype,
) flux.Flux(@TypeOf(p_LR.L), @TypeOf(v_LR.L)) {
    const DType = @TypeOf(p_LR.L);
    const VectorType = @TypeOf(v_LR.L);

    // Compute left right fluxes
    const flux_L = flux.convectiveFlux(p_LR.L, v_LR.L, rho_LR.L, H_LR.L, normal);
    const flux_R = flux.convectiveFlux(p_LR.R, v_LR.R, rho_LR.R, H_LR.R, normal);

    // Contra-variant velocities
    const u_L = v_LR.L.dot(normal);
    const u_R = v_LR.R.dot(normal);

    // Sound Speeds
    const c_L = eos.soundSpeedEnthalpy(H_LR.L, v_LR.L);
    const c_R = eos.soundSpeedEnthalpy(H_LR.R, v_LR.R);

    // Compute Pressure Star *
    const rho_bar = 0.5 * (rho_LR.L + rho_LR.R);
    const c_bar = 0.5 * (c_L + c_R);

    const p_pvrs = 0.5 * (p_LR.L + p_LR.R) - 0.5 * (u_R - u_L) * rho_bar * c_bar;
    const p_star = @max(0, p_pvrs);

    // Wave Speeds
    const q_L: DType = if (p_star > p_LR.L)
        @sqrt(1.0 + (eos.gamma + 1.0) / (2.0 * eos.gamma) * (p_star / p_LR.L - 1.0))
    else
        1.0;

    const q_R: DType = if (p_star > p_LR.R)
        @sqrt(1.0 + (eos.gamma + 1.0) / (2.0 * eos.gamma) * (p_star / p_LR.R - 1.0))
    else
        1.0;

    const S_L = u_L - c_L * q_L;
    const S_R = u_R + c_R * q_R;

    const S_L_term = rho_LR.L * (S_L - u_L);
    const S_R_term = rho_LR.R * (S_R - u_R);
    const S_star = (p_LR.R - p_LR.L + u_L * S_L_term - u_R * S_R_term) / (S_L_term - S_R_term);

    var flux_rho: DType = 0;
    var flux_rho_v: VectorType = .{};
    var flux_rho_E: DType = 0;

    if (S_L >= 0) {
        flux_rho = flux_L.flux_rho;
        flux_rho_v = flux_L.flux_rho_v;
        flux_rho_E = flux_L.flux_rho_E;
    } else if (S_L <= 0 and S_star >= 0) {
        const T_L = eos.temperatureEnthalpyTotal(H_LR.L, v_LR.L);
        const E_L = eos.energyTotal(T_L, v_LR.L);

        const rho_star = rho_LR.L * (S_L - u_L) / (S_L - S_star);
        const rho_v_star = v_LR.L.scale(rho_star).add(normal.scale(S_star - u_L));
        const rho_E_star = rho_star * (E_L +
            (S_star - u_L) * (S_star + p_LR.L / (rho_LR.L * (S_L - u_L))) / (eos.gamma - 1.0));

        flux_rho = flux_L.flux_rho + S_L * (rho_star - rho_LR.L);
        flux_rho_v = flux_L.flux_rho_v.add(rho_v_star.sub(v_LR.L.scale(rho_LR.L)).scale(S_L));
        flux_rho_E = flux_L.flux_rho_E + S_L * (rho_E_star - rho_LR.L * E_L);
    } else if (S_star <= 0 and S_R >= 0) {
        const T_R = eos.temperatureEnthalpyTotal(H_LR.R, v_LR.R);
        const E_R = eos.energyTotal(T_R, v_LR.R);

        const rho_star = rho_LR.R * (S_R - u_R) / (S_R - S_star);
        const rho_v_star = v_LR.R.scale(rho_star).add(normal.scale(S_star - u_R));
        const rho_E_star = rho_star * (E_R +
            (S_star - u_R) * (S_star + p_LR.R / (rho_LR.R * (S_R - u_R))) / (eos.gamma - 1.0));

        flux_rho = flux_R.flux_rho + S_R * (rho_star - rho_LR.R);
        flux_rho_v = flux_R.flux_rho_v.add(rho_v_star.sub(v_LR.R.scale(rho_LR.R)).scale(S_R));
        flux_rho_E = flux_R.flux_rho_E + S_R * (rho_E_star - rho_LR.R * E_R);
    } else {
        flux_rho = flux_R.flux_rho;
        flux_rho_v = flux_R.flux_rho_v;
        flux_rho_E = flux_R.flux_rho_E;
    }

    return .{ .flux_rho = flux_rho, .flux_rho_v = flux_rho_v, .flux_rho_E = flux_rho_E };
}
