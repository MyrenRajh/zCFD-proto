const std = @import("std");
const mp = @import("multiproc.zig");

pub fn computeGeometricGradientMatrix(system: anytype) void {
    const DType = @TypeOf(system.nodes.items(.centroid)[0].data[0]);

    // Increment grad matrices using face contributions
    for (0..system.n_faces) |i| {
        const i_owner = system.faces.items(.i_owner)[i];
        const i_neighbour = system.faces.items(.i_neighbour)[i];

        const owner_centroid = system.cells.items(.centroid)[i_owner];
        const neighbour_centroid = system.cells.items(.centroid)[i_neighbour];

        const r_of = neighbour_centroid.sub(owner_centroid);
        const d_of = r_of.mag();

        const w = 1.0 / std.math.pow(DType, d_of, 2.0);
        const grad_matrix_inc = r_of.outer(r_of).scale(w);

        const owner_grad_matrix = system.cells.items(.grad_matrix)[i_owner];
        const neighbour_grad_matrix = system.cells.items(.grad_matrix)[i_neighbour];

        system.cells.items(.grad_matrix)[i_owner] = owner_grad_matrix.add(grad_matrix_inc);
        system.cells.items(.grad_matrix)[i_neighbour] = neighbour_grad_matrix.add(grad_matrix_inc);
    }

    // Calculate inverse of grad matrices in internal cells
    for (0..system.n_cells) |i| {
        var grad_matrix = system.cells.items(.grad_matrix)[i];
        system.cells.items(.grad_matrix)[i] = grad_matrix.inv();
    }
}

pub fn kernelGradIncLeastSquares(
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
    const MatrixType = @TypeOf(system.cells.items(.grad_v)[0]);

    // Increment gradients
    for (thread_i_s..thread_i_f) |i_f| {
        // Get owner neighbour indices
        const i_o = system.faces.items(.i_owner)[i_f];
        const i_n = system.faces.items(.i_neighbour)[i_f];

        // Get geometric values
        const centroid_o = system.cells.items(.centroid)[i_o];
        const centroid_n = system.cells.items(.centroid)[i_n];

        // Calculate weight vector
        const r_on = centroid_n.sub(centroid_o);
        const d_on = r_on.mag();
        const weight_on = 1.0 / std.math.pow(DType, d_on, 2);

        // Increment grad_p
        const p_o = system.cells.items(.p)[i_o];
        const p_n = system.cells.items(.p)[i_n];

        const grad_p_inc = r_on.scale(weight_on * (p_n - p_o));

        VectorType.atomicAdd(&system.cells.items(.grad_p)[i_o], grad_p_inc);
        VectorType.atomicAdd(&system.cells.items(.grad_p)[i_n], grad_p_inc);

        // Increment grad_v
        const v_o = system.cells.items(.v)[i_o];
        const v_n = system.cells.items(.v)[i_n];

        const grad_v_inc = r_on.outer(v_n.sub(v_o)).scale(weight_on);

        MatrixType.atomicAdd(&system.cells.items(.grad_v)[i_o], grad_v_inc);
        MatrixType.atomicAdd(&system.cells.items(.grad_v)[i_n], grad_v_inc);

        // Increment grad_T
        const T_o = system.cells.items(.T)[i_o];
        const T_n = system.cells.items(.T)[i_n];

        const grad_T_inc = r_on.scale(weight_on * (T_n - T_o));

        VectorType.atomicAdd(&system.cells.items(.grad_T)[i_o], grad_T_inc);
        VectorType.atomicAdd(&system.cells.items(.grad_T)[i_n], grad_T_inc);
    }
}

pub fn kernelAssignGradLeastSquares(
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
        const grad_matrix = system.cells.items(.grad_matrix)[i_o];

        // Calculate grad_p
        const grad_p = system.cells.items(.grad_p)[i_o];
        system.cells.items(.grad_p)[i_o] = grad_matrix.mulVec(grad_p);

        // Calculate grad_v
        const grad_v = system.cells.items(.grad_v)[i_o];
        system.cells.items(.grad_v)[i_o] = grad_matrix.mul(grad_v);

        // Calculate grad_T
        const grad_T = system.cells.items(.grad_T)[i_o];
        system.cells.items(.grad_T)[i_o] = grad_matrix.mulVec(grad_T);
    }
}

pub fn kernelClearGradients(
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
    const MatrixType = @TypeOf(system.cells.items(.grad_v)[0]);

    for (thread_i_s..thread_i_f) |i_o| {
        system.cells.items(.grad_p)[i_o] = VectorType{};
        system.cells.items(.grad_v)[i_o] = MatrixType{};
        system.cells.items(.grad_T)[i_o] = VectorType{};
    }
}

pub fn computeGradientLeastSquares(system: anytype) !void {
    try mp.computeMultiProc(
        system,
        kernelClearGradients,
        .{},
        .{},
        system.n_cells,
        0,
    );

    try mp.computeMultiProc(
        system,
        kernelGradIncLeastSquares,
        .{},
        .{},
        system.n_faces,
        0,
    );

    try mp.computeMultiProc(
        system,
        kernelAssignGradLeastSquares,
        .{},
        .{},
        system.n_cells,
        0,
    );
}
