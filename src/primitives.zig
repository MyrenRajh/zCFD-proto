const nz = @import("numzig.zig");

pub fn CellNavierStokes(
    comptime DType: type,
    comptime dim: u3,
) type {
    const Vector = nz.SmallVector(DType, dim);
    const Matrix = nz.SmallMatrix(DType, dim, dim);

    return struct {
        // Geometric Quantities
        centroid: Vector = .{},
        volume: DType = 0,
        grad_matrix: Matrix = .{},

        // Indexing
        n_faces: usize = 0,
        i_start_face: usize = 0,

        // Variables Primitives
        p: DType = 0,
        v: Vector = .{},
        T: DType = 0,

        // Gradients Primitives
        grad_p: Vector = .{},
        grad_v: Matrix = .{},
        grad_T: Vector = .{},

        // Temporal Derivatives Conservatives
        ddt_rho: DType = 0,
        ddt_rho_v: Vector = .{},
        ddt_rho_E: DType = 0,

        dt: DType = 1,

        // MinMax Primitives
        minmax_p: [2]DType = .{ 0, 0 },
        minmax_v: [2]Vector = .{ .{}, .{} },
        minmax_T: [2]DType = .{ 0, 0 },

        // Gradient Limiters Primitives
        limiter_p: DType = 1,
        limiter_v: Vector = Vector.splat(1),
        limiter_T: DType = 1,

        // Variables Primitives
        p_old: DType = 0,
        v_old: Vector = .{},
        T_old: DType = 0,
    };
}

pub fn FaceNavierStokes(
    comptime DType: type,
    comptime dim: u3,
) type {
    const Vector = nz.SmallVector(DType, dim);

    return struct {
        // Geometric Quantities
        centroid: Vector = .{},
        area: DType = 0.0,
        normal: Vector = .{},
        tangent: Vector = .{},

        // Indexing
        n_nodes: usize = 0,
        i_start_node: usize = 0,

        i_owner: usize = 0,
        i_neighbour: usize = 0,
    };
}

pub fn NodeNavierStokes(
    comptime DType: type,
    comptime dim: u3,
) type {
    const Vector = nz.SmallVector(DType, dim);

    return struct {
        // Geometric Quantities
        centroid: Vector = .{},
    };
}
