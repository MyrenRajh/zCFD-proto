const std = @import("std");
const bnd = @import("boundary.zig");
const lim = @import("limiters.zig");

pub fn readCase(system: anytype, allocator: std.mem.Allocator) !void {
    try readMeshInfo(system, allocator);
    try readSystemControls(system, allocator);
    try readParameterControls(system, allocator);
    try readFluidProperties(system, allocator);

    for (0..system.n_nodes) |_| {
        try system.nodes.append(system.allocator, .{});
    }
    for (0..system.n_faces) |_| {
        try system.faces.append(system.allocator, .{});
    }
    for (0..system.n_cells + system.n_cells_dummy) |_| {
        try system.cells.append(system.allocator, .{});
    }
    for (0..system.n_boundary_patches) |_| {
        try system.boundaries.append(system.allocator, .{});
    }

    try readMeshNodeData(system, allocator);
    try readMeshFaceData(system, allocator);
    try readMeshCellData(system, allocator);
    try readMeshBoundaryData(system, allocator);

    var sum_face_n_nodes: usize = 0;
    for (system.faces.items(.n_nodes)) |n| {
        sum_face_n_nodes += n;
    }

    var sum_cell_n_faces: usize = 0;
    for (system.cells.items(.n_faces)) |n| {
        sum_cell_n_faces += n;
    }

    try system.i_face_nodes.appendNTimes(0, sum_face_n_nodes);
    try system.i_cell_faces.appendNTimes(0, sum_cell_n_faces);

    try readFileScalarInt(
        "case/mesh/face/i_nodes",
        &system.i_face_nodes.items,
        allocator,
    );
    try readFileScalarInt(
        "case/mesh/cell/i_faces",
        &system.i_cell_faces.items,
        allocator,
    );
    try readBoundaryConditions(system, allocator);

    // Read Initial conditions
    try readFileScalarFloat(
        "case/solution/0/p",
        &system.cells.items(.p)[0..system.n_cells],
        allocator,
    );

    try readFileVectorFloat(
        "case/solution/0/v",
        &system.cells.items(.v)[0..system.n_cells],
        allocator,
    );

    try readFileScalarFloat(
        "case/solution/0/T",
        &system.cells.items(.T)[0..system.n_cells],
        allocator,
    );

    // Fill other initial values in all cells
    for (0..system.n_cells) |i_o| {
        system.cells.items(.dt)[i_o] = system.dt;

        system.cells.items(.minmax_p)[i_o] = .{ system.cells.items(.p)[i_o], system.cells.items(.p)[i_o] };
        system.cells.items(.minmax_v)[i_o] = .{ system.cells.items(.v)[i_o], system.cells.items(.v)[i_o] };
        system.cells.items(.minmax_T)[i_o] = .{ system.cells.items(.T)[i_o], system.cells.items(.T)[i_o] };
    }
}

pub fn readFileScalarInt(
    path: anytype,
    container: anytype,
    allocator: std.mem.Allocator,
) !void {
    var file = try std.fs.cwd().openFile(
        path,
        .{ .mode = .read_only },
    );
    defer file.close();
    const file_size: usize = @intCast((try file.stat()).size);
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    try file.reader().readNoEof(buffer);
    var lines = std.mem.tokenizeAny(
        u8,
        buffer,
        "\n",
    );

    var i: usize = 0;

    while (lines.next()) |line| {
        const data = try std.fmt.parseInt(
            @TypeOf(container.*[0]),
            line,
            10,
        );
        container.*[i] = data;
        i += 1;
    }
}

pub fn readFileScalarFloat(
    path: anytype,
    container: anytype,
    allocator: std.mem.Allocator,
) !void {
    var file = try std.fs.cwd().openFile(
        path,
        .{ .mode = .read_only },
    );
    defer file.close();
    const file_size = (try file.stat()).size;
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    try file.reader().readNoEof(buffer);
    var lines = std.mem.tokenizeAny(
        u8,
        buffer,
        "\n",
    );

    var i: usize = 0;

    while (lines.next()) |line| {
        const data = try std.fmt.parseFloat(@TypeOf(container.*[0]), line);
        container.*[i] = data;
        i += 1;
    }
}

pub fn readFileVectorInt(
    path: anytype,
    container: anytype,
    allocator: std.mem.Allocator,
) !void {
    var file = try std.fs.cwd().openFile(
        path,
        .{ .mode = .read_only },
    );
    defer file.close();
    const file_size = (try file.stat()).size;
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    try file.reader().readNoEof(buffer);
    var lines = std.mem.tokenizeAny(
        u8,
        buffer,
        "\n",
    );

    var i: usize = 0;

    while (lines.next()) |line| {
        var temp_line = std.mem.tokenizeAny(
            u8,
            line,
            "\t",
        );

        var j: usize = 0;
        while (temp_line.next()) |str| {
            const data = try std.fmt.parseInt(
                @TypeOf(container.*[0].data[0]),
                str,
                10,
            );
            container.*[i].data[j] = data;
            j += 1;
        }
        i += 1;
    }
}

pub fn readFileVectorFloat(
    path: anytype,
    container: anytype,
    allocator: std.mem.Allocator,
) !void {
    var file = try std.fs.cwd().openFile(
        path,
        .{ .mode = .read_only },
    );
    defer file.close();
    const file_size = (try file.stat()).size;
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    try file.reader().readNoEof(buffer);
    var lines = std.mem.tokenizeAny(
        u8,
        buffer,
        "\n",
    );

    var i: usize = 0;

    while (lines.next()) |line| {
        var temp_line = std.mem.tokenizeAny(
            u8,
            line,
            "\t",
        );

        var j: usize = 0;
        while (temp_line.next()) |str| {
            const data = try std.fmt.parseFloat(
                @TypeOf(container.*[0].data[0]),
                str,
            );
            container.*[i].data[j] = data;
            j += 1;
        }
        i += 1;
    }
}

pub fn readFileEnum(
    path: anytype,
    container: anytype,
    allocator: std.mem.Allocator,
) !void {
    var file = try std.fs.cwd().openFile(
        path,
        .{ .mode = .read_only },
    );
    defer file.close();
    const file_size: usize = @intCast((try file.stat()).size);
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);
    try file.reader().readNoEof(buffer);
    var lines = std.mem.tokenizeAny(
        u8,
        buffer,
        "\n",
    );

    var i: usize = 0;

    while (lines.next()) |line| {
        const data = try std.fmt.parseInt(
            i64,
            line,
            10,
        );

        container.*[i] = @enumFromInt(data);
        i += 1;
    }
}

pub fn readMeshInfo(system: anytype, allocator: std.mem.Allocator) !void {
    var mesh_info = [_]usize{0} ** 7;

    try readFileScalarInt(
        "case/mesh/info",
        &mesh_info,
        allocator,
    );

    system.n_nodes = mesh_info[0];
    system.n_faces = mesh_info[1];
    system.n_cells = mesh_info[2];
    system.n_boundary_patches = mesh_info[3];
    system.n_faces_internal = mesh_info[4];
    system.n_faces_external = mesh_info[5];
    system.n_cells_dummy = mesh_info[6];
}

pub fn readSystemControls(system: anytype, allocator: std.mem.Allocator) !void {
    var info = [_]usize{0} ** 9;

    try readFileScalarInt("case/controls/system", &info, allocator);

    system.n_threads = info[0];
    system.save_every_n_steps = info[1];
    system.n_sim_steps = info[2];
    system.time_integration_scheme = @enumFromInt(info[3]);
    system.use_local_time_stepping = info[4] != 0;
    system.auto_calc_time_step = info[5] != 0;
    system.limiter_type = @enumFromInt(info[6]);
    system.conv_flux_scheme = @enumFromInt(info[7]);
    system.visc_flux_scheme = @enumFromInt(info[8]);
}

pub fn readParameterControls(system: anytype, allocator: std.mem.Allocator) !void {
    const DType = @TypeOf(system.dt);

    var info = [_]DType{0} ** 4;

    try readFileScalarFloat("case/controls/parameters", &info, allocator);

    system.dt = info[0];
    system.cfl = info[1];
    system.local_dt_C_factor = info[2];
    system.limiter_k = info[3];
}

pub fn readFluidProperties(system: anytype, allocator: std.mem.Allocator) !void {
    const DType = @TypeOf(system.dt);

    var info = [_]DType{0} ** 10;

    try readFileScalarFloat(
        "case/controls/properties",
        &info,
        allocator,
    );

    system.constitutive.R = info[0];
    system.constitutive.gamma = info[1];
    system.constitutive.Pr = info[2];
    system.constitutive.c_p = info[3];
    system.constitutive.c_v = info[4];
    system.constitutive.reference_p = info[5];
    system.constitutive.reference_v = info[6];
    system.constitutive.reference_T = info[7];
    system.constitutive.reference_mu = info[8];
    system.constitutive.reference_k = info[9];
}

pub fn readMeshNodeData(system: anytype, allocator: std.mem.Allocator) !void {
    try readFileVectorFloat(
        "case/mesh/node/centroid",
        &system.nodes.items(.centroid),
        allocator,
    );
}

pub fn readMeshFaceData(system: anytype, allocator: std.mem.Allocator) !void {
    try readFileScalarInt(
        "case/mesh/face/i_start_node",
        &system.faces.items(.i_start_node),
        allocator,
    );
    try readFileScalarInt(
        "case/mesh/face/n_nodes",
        &system.faces.items(.n_nodes),
        allocator,
    );
    try readFileScalarInt(
        "case/mesh/face/i_owner",
        &system.faces.items(.i_owner),
        allocator,
    );
    try readFileScalarInt(
        "case/mesh/face/i_neighbour",
        &system.faces.items(.i_neighbour),
        allocator,
    );
}

pub fn readMeshCellData(system: anytype, allocator: std.mem.Allocator) !void {
    try readFileScalarInt(
        "case/mesh/cell/i_start_face",
        &system.cells.items(.i_start_face),
        allocator,
    );
    try readFileScalarInt(
        "case/mesh/cell/n_faces",
        &system.cells.items(.n_faces),
        allocator,
    );
}

pub fn readMeshBoundaryData(system: anytype, allocator: std.mem.Allocator) !void {
    try readFileScalarInt(
        "case/mesh/boundary/i_start_face",
        &system.boundaries.items(.i_start_face),
        allocator,
    );

    try readFileScalarInt(
        "case/mesh/boundary/n_faces",
        &system.boundaries.items(.n_faces),
        allocator,
    );
}

pub fn readBoundaryConditions(system: anytype, allocator: std.mem.Allocator) !void {
    try readFileEnum(
        "case/controls/boundary/type",
        &system.boundaries.items(.type),
        allocator,
    );
    try readFileScalarFloat(
        "case/controls/boundary/p",
        &system.boundaries.items(.p),
        allocator,
    );
    try readFileVectorFloat(
        "case/controls/boundary/v",
        &system.boundaries.items(.v),
        allocator,
    );
    try readFileScalarFloat(
        "case/controls/boundary/T",
        &system.boundaries.items(.T),
        allocator,
    );
    try readFileScalarFloat(
        "case/controls/boundary/grad_p_normal",
        &system.boundaries.items(.grad_p_normal),
        allocator,
    );
    try readFileVectorFloat(
        "case/controls/boundary/grad_v_normal",
        &system.boundaries.items(.grad_v_normal),
        allocator,
    );
    try readFileScalarFloat(
        "case/controls/boundary/grad_T_normal",
        &system.boundaries.items(.grad_T_normal),
        allocator,
    );
}

pub fn writeSimData(
    system: anytype,
    allocator: std.mem.Allocator,
    step_no: usize,
) !void {
    // const DataType = @TypeOf(system.cells.items(.p)[0]);
    // const VectorType = @TypeOf(system.cells.items(.v)[0]);

    const cwd = std.fs.cwd();

    // Make solution dir
    const dir_name = try std.fmt.allocPrint(
        allocator,
        "case/solution/{d}",
        .{step_no},
    );
    defer allocator.free(dir_name);

    try cwd.makeDir(dir_name);

    // Make pressure file
    const file_p_name = try std.fmt.allocPrint(allocator, "case/solution/{d}/p", .{step_no});
    defer allocator.free(file_p_name);
    const file_p = try cwd.createFile(file_p_name, .{});
    defer file_p.close();

    // Make velocity file
    const file_v_name = try std.fmt.allocPrint(allocator, "case/solution/{d}/v", .{step_no});
    defer allocator.free(file_v_name);
    const file_v = try cwd.createFile(file_v_name, .{});
    defer file_v.close();

    // Make temperature file
    const file_T_name = try std.fmt.allocPrint(allocator, "case/solution/{d}/T", .{step_no});
    defer allocator.free(file_T_name);
    const file_T = try cwd.createFile(file_T_name, .{});
    defer file_T.close();

    // Make pressure gradient file
    const file_grad_p_name = try std.fmt.allocPrint(allocator, "case/solution/{d}/grad_p", .{step_no});
    defer allocator.free(file_grad_p_name);
    const file_grad_p = try cwd.createFile(file_grad_p_name, .{});
    defer file_grad_p.close();

    // Make velocity gradient file
    const file_grad_v_name = try std.fmt.allocPrint(allocator, "case/solution/{d}/grad_v", .{step_no});
    defer allocator.free(file_grad_v_name);
    const file_grad_v = try cwd.createFile(file_grad_v_name, .{});
    defer file_grad_v.close();

    // Make temperature gradient file
    const file_grad_T_name = try std.fmt.allocPrint(allocator, "case/solution/{d}/grad_T", .{step_no});
    defer allocator.free(file_grad_T_name);
    const file_grad_T = try cwd.createFile(file_grad_T_name, .{});
    defer file_grad_T.close();

    _ = try file_p.writeAll(std.mem.sliceAsBytes(system.cells.items(.p)[0..system.n_cells]));
    _ = try file_v.writeAll(std.mem.sliceAsBytes(system.cells.items(.v)[0..system.n_cells]));
    _ = try file_T.writeAll(std.mem.sliceAsBytes(system.cells.items(.T)[0..system.n_cells]));

    _ = try file_grad_p.writeAll(std.mem.sliceAsBytes(system.cells.items(.grad_p)[0..system.n_cells]));
    _ = try file_grad_v.writeAll(std.mem.sliceAsBytes(system.cells.items(.grad_v)[0..system.n_cells]));
    _ = try file_grad_T.writeAll(std.mem.sliceAsBytes(system.cells.items(.grad_T)[0..system.n_cells]));
}

pub fn writeGeomData(
    system: anytype,
    allocator: std.mem.Allocator,
) !void {
    const cwd = std.fs.cwd();

    // Cell Centroid
    const file_cell_centroid_name = try std.fmt.allocPrint(
        allocator,
        "case/mesh/cell/centroid",
        .{},
    );
    defer allocator.free(file_cell_centroid_name);

    const file_cell_centroid = try cwd.createFile(
        file_cell_centroid_name,
        .{},
    );
    defer file_cell_centroid.close();

    const file_cell_centroid_writer = file_cell_centroid.writer();

    // Cell volume
    const file_cell_volume_name = try std.fmt.allocPrint(
        allocator,
        "case/mesh/cell/volume",
        .{},
    );
    defer allocator.free(file_cell_volume_name);

    const file_cell_volume = try cwd.createFile(
        file_cell_volume_name,
        .{},
    );
    defer file_cell_volume.close();

    const file_cell_volume_writer = file_cell_volume.writer();

    for (0..system.n_cells) |i_o| {
        const centroid_o = system.cells.items(.centroid)[i_o];
        const volume_o = system.cells.items(.volume)[i_o];

        const str_centroid = try std.fmt.allocPrint(
            allocator,
            "{:.8}\t{:.8}\t{:.8}\n",
            .{ centroid_o.data[0], centroid_o.data[1], centroid_o.data[2] },
        );

        const str_volume = try std.fmt.allocPrint(
            allocator,
            "{:.8}\n",
            .{volume_o},
        );

        _ = try file_cell_centroid_writer.write(str_centroid);
        _ = try file_cell_volume_writer.write(str_volume);
    }

    // Face Centroid
    const file_face_centroid_name = try std.fmt.allocPrint(
        allocator,
        "case/mesh/face/centroid",
        .{},
    );
    defer allocator.free(file_face_centroid_name);

    const file_face_centroid = try cwd.createFile(
        file_face_centroid_name,
        .{},
    );
    defer file_face_centroid.close();

    const file_face_centroid_writer = file_face_centroid.writer();

    // Face area
    const file_face_area_name = try std.fmt.allocPrint(
        allocator,
        "case/mesh/face/area",
        .{},
    );
    defer allocator.free(file_face_area_name);

    const file_face_area = try cwd.createFile(
        file_face_area_name,
        .{},
    );
    defer file_face_area.close();

    const file_face_area_writer = file_face_area.writer();

    // Face normal
    const file_face_normal_name = try std.fmt.allocPrint(
        allocator,
        "case/mesh/face/normal",
        .{},
    );
    defer allocator.free(file_face_normal_name);

    const file_face_normal = try cwd.createFile(
        file_face_normal_name,
        .{},
    );
    defer file_face_normal.close();

    const file_face_normal_writer = file_face_normal.writer();

    for (0..system.n_faces) |i_f| {
        const centroid_f = system.faces.items(.centroid)[i_f];

        const area = system.faces.items(.area)[i_f];
        const normal = system.faces.items(.normal)[i_f];

        const str_centroid = try std.fmt.allocPrint(
            allocator,
            "{:.8}\t{:.8}\t{:.8}\n",
            .{ centroid_f.data[0], centroid_f.data[1], centroid_f.data[2] },
        );

        const str_area = try std.fmt.allocPrint(
            allocator,
            "{:.8}\n",
            .{area},
        );

        const str_normal = try std.fmt.allocPrint(
            allocator,
            "{:.8}\t{:.8}\t{:.8}\n",
            .{ normal.data[0], normal.data[1], normal.data[2] },
        );

        _ = try file_face_centroid_writer.write(str_centroid);
        _ = try file_face_area_writer.write(str_area);
        _ = try file_face_normal_writer.write(str_normal);
    }
}
