const std = @import("std");

test "comptime test" {
    var single_threaded_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer single_threaded_arena.deinit();

    var thread_safe_arena: std.heap.ThreadSafeAllocator = .{
        .child_allocator = single_threaded_arena.allocator(),
    };
    const thread_allocator = thread_safe_arena.allocator();

    var vals = try thread_allocator.alloc(f64, 10);
    vals[0] = 1.0;

    var i: usize = 0;
    for (vals) |_| {
        std.debug.print("{} \n", .{(&vals).*[i]});
        i += 1;
    }
}
