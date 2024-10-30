const std = @import("std");
const fvm = @import("fvm_system.zig");
const prim = @import("primitives.zig");

const const_law = @import("constitutive.zig");

const DType = f64;
const dim = 3;
const FVMSystem = fvm.FVMSystem(
    prim.CellNavierStokes(DType, dim),
    prim.FaceNavierStokes(DType, dim),
    prim.NodeNavierStokes(DType, dim),
    const_law.PerfectGas(
        DType,
        // true,
        // true,
    ),
);

pub fn main() !void {
    // General Allocator
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    // Thread Allocator
    var single_threaded_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer single_threaded_arena.deinit();

    var thread_safe_arena: std.heap.ThreadSafeAllocator = .{
        .child_allocator = single_threaded_arena.allocator(),
    };
    const thread_allocator = thread_safe_arena.allocator();

    var system = FVMSystem.init(allocator, thread_allocator);
    defer system.deinit();

    try system.initCase();
    try system.solveSystem();
}
