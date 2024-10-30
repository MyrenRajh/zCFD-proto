const std = @import("std");

// Generic retrieval functions
pub fn getVariables(system: anytype, i_o: usize) struct {
    p: @TypeOf(system.cells.items(.p)[0]),
    v: @TypeOf(system.cells.items(.v)[0]),
    T: @TypeOf(system.cells.items(.T)[0]),
} {
    return .{
        .p = system.cells.items(.p)[i_o],
        .v = system.cells.items(.v)[i_o],
        .T = system.cells.items(.T)[i_o],
    };
}

pub fn getGradients(system: anytype, i_o: usize) struct {
    grad_p: @TypeOf(system.cells.items(.grad_p)[0]),
    grad_v: @TypeOf(system.cells.items(.grad_v)[0]),
    grad_T: @TypeOf(system.cells.items(.grad_T)[0]),
} {
    return .{
        .grad_p = system.cells.items(.grad_p)[i_o],
        .grad_v = system.cells.items(.grad_v)[i_o],
        .grad_T = system.cells.items(.grad_T)[i_o],
    };
}

pub fn getLimiters(system: anytype, i_o: usize) struct {
    limiter_p: @TypeOf(system.cells.items(.limiter_p)[0]),
    limiter_v: @TypeOf(system.cells.items(.limiter_v)[0]),
    limiter_T: @TypeOf(system.cells.items(.limiter_T)[0]),
} {
    return .{
        .limiter_p = system.cells.items(.limiter_p)[i_o],
        .limiter_v = system.cells.items(.limiter_v)[i_o],
        .limiter_T = system.cells.items(.limiter_T)[i_o],
    };
}

pub fn getMinMax(system: anytype, i_o: usize) struct {
    minmax_p: @TypeOf(system.cells.items(.minmax_p)[0]),
    minmax_v: @TypeOf(system.cells.items(.minmax_v)[0]),
    minmax_T: @TypeOf(system.cells.items(.minmax_T)[0]),
} {
    return .{
        .minmax_p = system.cells.items(.minmax_p)[i_o],
        .minmax_v = system.cells.items(.minmax_v)[i_o],
        .minmax_T = system.cells.items(.minmax_T)[i_o],
    };
}
