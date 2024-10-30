const std = @import("std");

pub fn computeMultiProc(
    system: anytype,
    kernel: anytype,
    kernel_args: anytype,
    comptime comptime_kernel_args: anytype,
    total_work: usize,
    thread_i_s_init: usize,
) !void {
    const thread_pool = try system.thread_allocator.alloc(std.Thread, system.n_threads);
    defer system.thread_allocator.free(thread_pool);

    const eq_work = @divFloor(total_work, system.n_threads);
    const rem_work = total_work % system.n_threads;

    var i: usize = 0;
    var thread_i_s: usize = thread_i_s_init;

    for (thread_pool) |*thread| {
        const ex_work: usize = if (i < rem_work) 1 else 0;
        const thread_i_f = thread_i_s + eq_work + ex_work;

        thread.* = try std.Thread.spawn(
            .{},
            kernel,
            .{ system, kernel_args, comptime_kernel_args, thread_i_s, thread_i_f, i },
        );

        i += 1;
        thread_i_s += eq_work + ex_work;
    }

    for (thread_pool) |thread| {
        thread.join();
    }
}
