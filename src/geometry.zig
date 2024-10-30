const std = @import("std");

pub fn triSurfaceVector(r_1: anytype, r_2: anytype, r_3: anytype) @TypeOf(r_1) {
    return (r_2.sub(r_1)).cross(r_3.sub(r_1)).scale(1.0 / 2.0);
}

pub fn triCentroid(r_1: anytype, r_2: anytype, r_3: anytype) @TypeOf(r_1) {
    return r_1.add(r_2).add(r_3).scale(1.0 / 3.0);
}

pub fn pyrCentroid(geom_centroid: anytype, face_centroid: anytype) @TypeOf(geom_centroid) {
    return face_centroid.scale(3.0 / 4.0).add(geom_centroid.scale(1.0 / 4.0));
}

pub fn pyrVolume(
    geom_centroid: anytype,
    face_centroid: anytype,
    surface_vector: anytype,
) @TypeOf(geom_centroid.data[0]) {
    return @abs((face_centroid.sub(geom_centroid)).dot(surface_vector)) / 3.0;
}

pub fn localNormalBasis(normal: anytype, tangent: anytype) struct {
    normal: @TypeOf(normal),
    tangent: @TypeOf(normal),
    bitangent: @TypeOf(normal),
    transform_mat: @TypeOf(normal.outer(normal)),
    transform_mat_T: @TypeOf(normal.outer(normal)),
} {
    const VectorType = @TypeOf(normal);
    const MatrixType = @TypeOf(normal.outer(normal));

    var bitangent: VectorType = .{};
    var transform_mat_T: MatrixType = .{};

    if (normal.data.len == 2) {
        transform_mat_T.data[0] = normal.data;
        transform_mat_T.data[1] = tangent.data;
    } else {
        bitangent = normal.cross(tangent);
        transform_mat_T.data[0] = normal.data;
        transform_mat_T.data[1] = tangent.data;
        transform_mat_T.data[2] = bitangent.data;
    }
    const transform_mat = transform_mat_T.transpose();

    return .{
        .normal = normal,
        .tangent = tangent,
        .bitangent = bitangent,
        .transform_mat = transform_mat,
        .transform_mat_T = transform_mat_T,
    };
}

pub fn localGeometry(centroid_o: anytype, centroid_f: anytype, local_basis: anytype) struct {
    r_of: @TypeOf(centroid_o),
    r_oo_n: @TypeOf(centroid_o),
    r_of_n: @TypeOf(centroid_o),
} {
    const normal = local_basis.normal;
    const r_of = centroid_f.sub(centroid_o);

    const centroid_o_n = centroid_f.sub(normal.scale(r_of.dot(normal)));
    const r_oo_n = centroid_o_n.sub(centroid_o);
    const r_of_n = centroid_f.sub(centroid_o_n);

    return .{ .r_of = r_of, .r_oo_n = r_oo_n, .r_of_n = r_of_n };
}

pub fn computeFaceGeometry(system: anytype) void {
    const Vector = @TypeOf(system.nodes.items(.centroid)[0]);
    const DType = @TypeOf(system.nodes.items(.centroid)[0].data[0]);

    // Calculate face centroids and surface vectors
    for (0..system.n_faces) |i| {
        // Get face number of nodes and start indices of nodes
        const n_nodes = system.faces.items(.n_nodes)[i];
        const i_start_node = system.faces.items(.i_start_node)[i];

        // Calculate Tangent
        const i_n0 = system.i_face_nodes.items[i_start_node];
        const i_n1 = system.i_face_nodes.items[i_start_node + 1];
        const centroid_n0 = system.nodes.items(.centroid)[i_n0];
        const centroid_n1 = system.nodes.items(.centroid)[i_n1];
        const tangent = centroid_n1.sub(centroid_n0).unit();

        // Calculate geometric centroid and tangent
        var geom_centroid: Vector = .{};
        for (0..n_nodes) |j| {
            const k = system.i_face_nodes.items[i_start_node + j];
            const node_centroid = system.nodes.items(.centroid)[k];
            geom_centroid = geom_centroid.add(node_centroid);
        }
        geom_centroid = geom_centroid.scale(1.0 / @as(DType, @floatFromInt(n_nodes)));

        // Calculate area, surface vector, centroid
        var area: DType = 0.0;
        var surface_vector: Vector = .{};
        var centroid: Vector = .{};

        for (0..n_nodes) |j| {
            const k = system.i_face_nodes.items[i_start_node + j];
            const l = system.i_face_nodes.items[i_start_node + (j + 1) % n_nodes];

            const r_1 = system.nodes.items(.centroid)[k];
            const r_2 = system.nodes.items(.centroid)[l];

            const tri_surface_vector = triSurfaceVector(r_1, r_2, geom_centroid);
            const tri_centroid = triCentroid(r_1, r_2, geom_centroid);
            const tri_area = tri_surface_vector.mag();

            area += tri_area;
            surface_vector = surface_vector.add(tri_surface_vector.unit());
            centroid = centroid.add(tri_centroid.scale(tri_area));
        }

        surface_vector = surface_vector.scale(1.0 / @as(DType, @floatFromInt(n_nodes)) * area);
        centroid = centroid.scale(1.0 / area);

        const normal = surface_vector.unit();

        // Assign centroid and surface vector
        system.faces.items(.centroid)[i] = centroid;
        system.faces.items(.area)[i] = area;
        system.faces.items(.normal)[i] = normal;
        system.faces.items(.tangent)[i] = tangent;
    }
}

pub fn computeCellGeometry(system: anytype) void {
    const Vector = @TypeOf(system.nodes.items(.centroid)[0]);
    const DType = @TypeOf(system.nodes.items(.centroid)[0].data[0]);

    // Calculate cell centroids and volumes
    for (0..system.n_cells) |i| {
        // Get cell number of faces and start indice of face
        const n_faces = system.cells.items(.n_faces)[i];
        const i_start_face = system.cells.items(.i_start_face)[i];

        // Calculate geometric centroid
        var geom_centroid: Vector = .{};
        for (0..n_faces) |j| {
            const k = system.i_cell_faces.items[i_start_face + j];
            const face_centroid = system.faces.items(.centroid)[k];
            geom_centroid = geom_centroid.add(face_centroid);
        }
        geom_centroid = geom_centroid.scale(1.0 / @as(DType, @floatFromInt(n_faces)));

        // Calculate volume and centroid
        var volume: DType = 0;
        var centroid: Vector = .{};

        for (0..n_faces) |j| {
            const k = system.i_cell_faces.items[i_start_face + j];

            const face_centroid = system.faces.items(.centroid)[k];
            const face_area = system.faces.items(.area)[k];
            const face_normal = system.faces.items(.normal)[k];
            const face_surface_vector = face_normal.scale(face_area);

            const pyr_centroid = pyrCentroid(geom_centroid, face_centroid);
            const pyr_volume = pyrVolume(
                geom_centroid,
                face_centroid,
                face_surface_vector,
            );

            volume += pyr_volume;
            centroid = centroid.add(pyr_centroid.scale(pyr_volume));
        }

        centroid = centroid.scale(1.0 / volume);

        // Assign the volume and centroid
        system.cells.items(.volume)[i] = volume;
        system.cells.items(.centroid)[i] = centroid;
    }

    // Calculate dummy cell centroids and volumes and assign as neighbour
    for (0..system.n_faces_external) |i| {
        const i_f = system.n_faces_internal + i;
        const i_n = system.n_cells + i;
        const i_o = system.faces.items(.i_owner)[i_f];

        // Get face an owner cell quantities
        const centroid_f = system.faces.items(.centroid)[i_f];
        const volume_o = system.cells.items(.volume)[i_o];

        // // Use face centroid as neighbour centroid
        const centroid_n = centroid_f;

        // Assign neighbour centroid and volume and as neighbour
        system.faces.items(.i_neighbour)[i_f] = i_n;
        system.cells.items(.centroid)[i_n] = centroid_n;
        system.cells.items(.volume)[i_n] = volume_o;
    }
}
