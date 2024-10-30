const std = @import("std");

pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{
        .name = "zCFDproto",
        .root_source_file = b.path("src/main.zig"),
        .target = b.host,
        .optimize = .ReleaseFast,
    });

    b.installArtifact(exe);
}
