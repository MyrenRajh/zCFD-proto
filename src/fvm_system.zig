const std = @import("std");
const geom = @import("geometry.zig");
const rw = @import("read_write.zig");
const grad = @import("gradient.zig");
const lim = @import("limiters.zig");
const bnd = @import("boundary.zig");
const flux = @import("flux.zig");
const time = @import("temporal.zig");

pub fn FVMSystem(
    comptime Cell: type,
    comptime Face: type,
    comptime Node: type,
    comptime ConstitutiveLaw: type,
) type {
    const DType = std.meta.FieldType(Cell, .p);
    const VectorType = std.meta.FieldType(Cell, .v);
    const BoundaryInfoType = bnd.BoundaryInfo(DType, VectorType);

    return struct {
        // Allocator
        allocator: std.mem.Allocator,
        thread_allocator: std.mem.Allocator,

        // Mesh Info
        n_nodes: usize = 0,
        n_faces: usize = 0,
        n_cells: usize = 0,
        n_boundary_patches: usize = 0,
        n_faces_internal: usize = 0,
        n_faces_external: usize = 0,
        n_cells_dummy: usize = 0,

        // Field Structs
        cells: std.MultiArrayList(Cell),
        faces: std.MultiArrayList(Face),
        nodes: std.MultiArrayList(Node),
        boundaries: std.MultiArrayList(BoundaryInfoType),

        // Dynamic Field Connectivity
        i_face_nodes: std.ArrayList(usize),
        i_cell_faces: std.ArrayList(usize),

        // Thermo-Physical Properties
        constitutive: ConstitutiveLaw = .{},

        // System Controls
        n_threads: usize = 16,
        save_every_n_steps: usize = 1,
        n_sim_steps: usize = 100,

        // Time Controls
        time_integration_scheme: time.TimeIntegrationSchemes = .exp_multi_rk_1_3,
        use_local_time_stepping: bool = true,
        auto_calc_time_step: bool = false,

        dt: DType = 1.0,
        cfl: DType = 0.5,
        local_dt_C_factor: DType = 2.0,

        // Limiter Controls
        limiter_type: lim.LimiterTypes = .venkatakrishnan,
        limiter_k: DType = 1.0,

        // Flux controls
        conv_flux_scheme: flux.ConvectiveFluxScheme = .riemann_hllc,
        visc_flux_scheme: flux.ViscousFluxScheme = .interp_grad_linear,

        // Functions
        pub fn init(allocator: std.mem.Allocator, thread_allocator: std.mem.Allocator) @This() {
            return .{
                .cells = std.MultiArrayList(Cell){},
                .faces = std.MultiArrayList(Face){},
                .nodes = std.MultiArrayList(Node){},
                .boundaries = std.MultiArrayList(BoundaryInfoType){},

                .i_face_nodes = std.ArrayList(usize).init(allocator),
                .i_cell_faces = std.ArrayList(usize).init(allocator),

                .allocator = allocator,
                .thread_allocator = thread_allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.cells.deinit(self.allocator);
            self.faces.deinit(self.allocator);
            self.nodes.deinit(self.allocator);
            self.boundaries.deinit(self.allocator);

            self.i_face_nodes.deinit();
            self.i_cell_faces.deinit();
        }

        pub fn initCase(self: *@This()) !void {
            std.debug.print("Initialising System ... ", .{});

            try rw.readCase(self, self.allocator);

            geom.computeFaceGeometry(self);
            geom.computeCellGeometry(self);
            grad.computeGeometricGradientMatrix(self);

            // Write geometry data
            try rw.writeGeomData(self, self.allocator);

            // Satisfy BC initial field and Initial Gradients
            try self.computePostStep();

            // If using local stepping get first step
            if (self.use_local_time_stepping) {
                try time.computeLocalTimeStep(self);
            } else if (self.auto_calc_time_step) {
                try time.computeLocalTimeStep(self);
                const global_min_dt = try time.computeMinTimeStep(self);
                try time.computeAssignTimeStep(self, global_min_dt);
            }

            std.debug.print("Success! \n", .{});
        }

        pub fn computeFluxes(self: *@This()) !void {
            switch (self.conv_flux_scheme) {
                .riemann_hllc => switch (self.visc_flux_scheme) {
                    .interp_grad_linear => try flux.computeFaceFluxes(self, .riemann_hllc, .interp_grad_linear),
                    .none => try flux.computeFaceFluxes(self, .riemann_hllc, .none),
                },
                .riemann_roe => switch (self.visc_flux_scheme) {
                    .interp_grad_linear => try flux.computeFaceFluxes(self, .riemann_roe, .interp_grad_linear),
                    .none => try flux.computeFaceFluxes(self, .riemann_roe, .none),
                },
                .none => switch (self.visc_flux_scheme) {
                    .interp_grad_linear => try flux.computeFaceFluxes(self, .none, .interp_grad_linear),
                    .none => {},
                },
            }
        }

        pub fn computePostStep(self: *@This()) !void {
            try grad.computeGradientLeastSquares(self);

            try bnd.computeBCs(self);

            switch (self.limiter_type) {
                .venkatakrishnan => {
                    try lim.computeMinMaxVariables(self);
                    try lim.computeUnstructuredLimiters(self, .venkatakrishnan);
                },
                .barth_jeperson => {
                    try lim.computeMinMaxVariables(self);
                    try lim.computeUnstructuredLimiters(self, .venkatakrishnan);
                },
            }
        }

        pub fn timeStep(self: *@This(), step_no: usize) !void {
            switch (self.time_integration_scheme) {
                .exp_euler => try time.computeExplicitTimeIntegration(self, .exp_euler),
                .exp_multi_rk_1_3 => try time.computeExplicitTimeIntegration(self, .exp_multi_rk_1_3),
                .exp_multi_rk_1_4 => try time.computeExplicitTimeIntegration(self, .exp_multi_rk_1_4),
                .exp_multi_rk_1_5 => try time.computeExplicitTimeIntegration(self, .exp_multi_rk_1_5),
                .exp_multi_rk_2_3 => try time.computeExplicitTimeIntegration(self, .exp_multi_rk_2_3),
                .exp_multi_rk_2_4 => try time.computeExplicitTimeIntegration(self, .exp_multi_rk_2_4),
                .exp_multi_rk_2_5 => try time.computeExplicitTimeIntegration(self, .exp_multi_rk_2_5),
            }

            if (@rem(step_no, self.save_every_n_steps) == 0) {
                try rw.writeSimData(self, self.allocator, step_no);
            }

            if (self.use_local_time_stepping) {
                try time.computeLocalTimeStep(self);
            } else if (self.auto_calc_time_step) {
                try time.computeLocalTimeStep(self);
                const global_min_dt = try time.computeMinTimeStep(self);
                try time.computeAssignTimeStep(self, global_min_dt);
            }
        }

        pub fn solveSystem(self: *@This()) !void {
            for (1..self.n_sim_steps + 1) |n_t| {
                std.debug.print("Simulating Step : {} \n", .{n_t});
                try self.timeStep(n_t);
            }
        }
    };
}
