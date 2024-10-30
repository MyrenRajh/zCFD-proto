const std = @import("std");

pub fn SmallVector(
    comptime DataType: type,
    comptime dim: u3,
) type {
    return struct {
        data: [dim]DataType = .{0} ** dim,

        pub fn splat(scalar: DataType) @This() {
            return .{ .data = .{scalar} ** dim };
        }

        pub fn clone(vec: @This()) @This() {
            return .{ .data = vec.data };
        }

        pub fn add(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = vec_1.data[i] + vec_2.data[i];
            }
            return out;
        }

        pub fn sub(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = vec_1.data[i] - vec_2.data[i];
            }
            return out;
        }

        pub fn mulElem(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = vec_1.data[i] * vec_2.data[i];
            }
            return out;
        }

        pub fn divElem(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = vec_1.data[i] / vec_2.data[i];
            }
            return out;
        }

        pub fn scale(vec: @This(), scalar: DataType) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = vec.data[i] * scalar;
            }
            return out;
        }

        pub fn dot(vec_1: @This(), vec_2: @This()) DataType {
            var out: DataType = 0;
            for (0..dim) |i| {
                out += vec_1.data[i] * vec_2.data[i];
            }
            return out;
        }

        pub fn mag(vec: @This()) DataType {
            const out: DataType = @sqrt(dot(vec, vec));
            return out;
        }

        pub fn unit(vec: @This()) @This() {
            const out: @This() = vec.scale(1.0 / vec.mag());
            return out;
        }

        pub fn abs(vec: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = @abs(vec.data[i]);
            }
            return out;
        }

        pub fn sum(vec: @This()) DataType {
            var out: DataType = 0;
            for (0..dim) |i| {
                out += vec.data[i];
            }
            return out;
        }

        pub fn addScalar(vec: @This(), scalar: DataType) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = vec.data[i] + scalar;
            }
            return out;
        }

        pub fn cross(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};

            out.data[0] = vec_1.data[1] * vec_2.data[2] - vec_1.data[2] * vec_2.data[1];
            out.data[1] = vec_1.data[2] * vec_2.data[0] - vec_1.data[0] * vec_2.data[2];
            out.data[2] = vec_1.data[0] * vec_2.data[1] - vec_1.data[1] * vec_2.data[0];

            return out;
        }

        pub fn outer(vec_1: @This(), vec_2: anytype) SmallMatrix(
            DataType,
            @typeInfo(@TypeOf(vec_1.data)).Array.len,
            @typeInfo(@TypeOf(vec_2.data)).Array.len,
        ) {
            const rows = @typeInfo(@TypeOf(vec_1.data)).Array.len;
            const cols = @typeInfo(@TypeOf(vec_2.data)).Array.len;

            var out: SmallMatrix(DataType, rows, cols) = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i][j] = vec_1.data[i] * vec_2.data[j];
                }
            }

            return out;
        }

        pub fn maxElem(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = @max(vec_1.data[i], vec_2.data[i]);
            }
            return out;
        }

        pub fn minElem(vec_1: @This(), vec_2: @This()) @This() {
            var out: @This() = .{};
            for (0..dim) |i| {
                out.data[i] = @min(vec_1.data[i], vec_2.data[i]);
            }
            return out;
        }

        pub fn atomicAdd(vec_1: *@This(), vec_2: @This()) void {
            for (0..dim) |i| {
                _ = @atomicRmw(DataType, &vec_1.data[i], .Add, vec_2.data[i], .acq_rel);
            }
        }

        pub fn atomicSub(vec_1: *@This(), vec_2: @This()) void {
            for (0..dim) |i| {
                _ = @atomicRmw(DataType, &vec_1.data[i], .Sub, vec_2.data[i], .acq_rel);
            }
        }

        pub fn atomicMinElem(vec_1: *@This(), vec_2: @This()) void {
            for (0..dim) |i| {
                _ = @atomicRmw(DataType, &vec_1.data[i], .Min, vec_2.data[i], .acq_rel);
            }
        }

        pub fn atomicMaxElem(vec_1: *@This(), vec_2: @This()) void {
            for (0..dim) |i| {
                _ = @atomicRmw(DataType, &vec_1.data[i], .Max, vec_2.data[i], .acq_rel);
            }
        }
    };
}

pub fn SmallMatrix(
    comptime DataType: type,
    comptime rows: u3,
    comptime cols: u3,
) type {
    return struct {
        data: [rows][cols]DataType = .{.{0} ** cols} ** rows,

        pub fn splat(scalar: DataType) @This() {
            return .{ .data = .{.{scalar} ** cols} ** rows };
        }

        pub fn clone(mat: @This()) @This() {
            return .{ .data = mat.data };
        }

        pub fn identity() @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                out[i][i] = 1;
            }

            return out;
        }

        pub fn diagSplat(scalar: DataType) @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                out.data[i][i] = scalar;
            }

            return out;
        }

        pub fn trace(mat: @This()) DataType {
            var out: DataType = 0;
            for (0..rows) |i| {
                out += mat.data[i][i];
            }
            return out;
        }

        pub fn add(mat_1: @This(), mat_2: @This()) @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i][j] = mat_1.data[i][j] + mat_2.data[i][j];
                }
            }

            return out;
        }

        pub fn sub(mat_1: @This(), mat_2: @This()) @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i][j] = mat_1.data[i][j] - mat_2.data[i][j];
                }
            }

            return out;
        }

        pub fn mulElem(mat_1: @This(), mat_2: @This()) @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i][j] = mat_1.data[i][j] * mat_2.data[i][j];
                }
            }

            return out;
        }

        pub fn divElem(mat_1: @This(), mat_2: @This()) @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i][j] = mat_1.data[i][j] / mat_2.data[i][j];
                }
            }

            return out;
        }

        pub fn scale(mat_1: @This(), scalar: DataType) @This() {
            var out: @This() = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i][j] = mat_1.data[i][j] * scalar;
                }
            }

            return out;
        }

        pub fn transpose(mat: @This()) SmallMatrix(
            DataType,
            cols,
            rows,
        ) {
            var out: SmallMatrix(DataType, cols, rows) = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[j][i] = mat.data[i][j];
                }
            }

            return out;
        }

        pub fn mulVec(mat: @This(), vec: anytype) SmallVector(
            DataType,
            rows,
        ) {
            var out: SmallVector(DataType, rows) = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i] += mat.data[i][j] * vec.data[j];
                }
            }

            return out;
        }

        pub fn transposeMulVec(mat: @This(), vec: anytype) SmallVector(
            DataType,
            rows,
        ) {
            var out: SmallVector(DataType, rows) = .{};

            for (0..rows) |i| {
                for (0..cols) |j| {
                    out.data[i] += mat.data[i][j] * vec.data[j];
                }
            }

            return out;
        }

        pub fn mul(mat_1: @This(), mat_2: @This()) SmallMatrix(
            DataType,
            @typeInfo(@TypeOf(mat_1.data)).Array.len,
            @typeInfo(@TypeOf(mat_1.data[0])).Array.len,
        ) {
            const out_rows = @typeInfo(@TypeOf(
                mat_1.data,
            )).Array.len;

            const out_cols = @typeInfo(@TypeOf(
                mat_2.data[0],
            )).Array.len;

            const in_dims = @typeInfo(@TypeOf(
                mat_1.data[0],
            )).Array.len;

            var out: SmallMatrix(
                DataType,
                out_rows,
                out_cols,
            ) = .{};

            for (0..out_rows) |i| {
                for (0..out_cols) |j| {
                    for (0..in_dims) |k| {
                        out.data[i][j] += mat_1.data[i][k] * mat_2.data[k][j];
                    }
                }
            }

            return out;
        }

        pub fn inv(mat: @This()) @This() {
            const eff_dim = @intFromBool(rows == cols) * rows;

            switch (eff_dim) {
                2 => {
                    return inv2(mat);
                },
                3 => {
                    return inv3(mat);
                },
                else => {
                    return .{};
                },
            }
        }

        fn inv2(mat: SmallMatrix(DataType, 2, 2)) SmallMatrix(
            DataType,
            2,
            2,
        ) {
            var out: SmallMatrix(DataType, 2, 2) = .{};
            const inv_det = 1 /
                (mat.data[0][0] * mat.data[1][1] -
                mat.data[0][1] * mat.data[1][0]);

            out.data[0][0] = inv_det * mat.data[1][1];
            out.data[1][1] = inv_det * mat.data[0][0];

            out.data[0][1] = -inv_det * mat.data[0][1];
            out.data[1][0] = -inv_det * mat.data[1][0];
            return out;
        }

        fn inv3(mat: SmallMatrix(DataType, 3, 3)) SmallMatrix(
            DataType,
            3,
            3,
        ) {
            var out: SmallMatrix(DataType, 3, 3) = .{};

            const A = (mat.data[1][1] * mat.data[2][2] - mat.data[1][2] * mat.data[2][1]);
            const B = -(mat.data[1][0] * mat.data[2][2] - mat.data[1][2] * mat.data[2][0]);
            const C = (mat.data[1][0] * mat.data[2][1] - mat.data[1][1] * mat.data[2][0]);
            const D = -(mat.data[0][1] * mat.data[2][2] - mat.data[0][2] * mat.data[2][1]);
            const E = (mat.data[0][0] * mat.data[2][2] - mat.data[0][2] * mat.data[2][0]);
            const F = -(mat.data[0][0] * mat.data[2][1] - mat.data[0][1] * mat.data[2][0]);
            const G = (mat.data[0][1] * mat.data[1][2] - mat.data[0][2] * mat.data[1][1]);
            const H = -(mat.data[0][0] * mat.data[1][2] - mat.data[0][2] * mat.data[1][0]);
            const I = (mat.data[0][0] * mat.data[1][1] - mat.data[0][1] * mat.data[1][0]);

            const inv_det = 1 /
                (mat.data[0][0] * A + mat.data[1][1] * B + mat.data[2][2] * C);

            out.data[0][0] = inv_det * A;
            out.data[1][0] = inv_det * B;
            out.data[2][0] = inv_det * C;
            out.data[0][1] = inv_det * D;
            out.data[1][1] = inv_det * E;
            out.data[2][1] = inv_det * F;
            out.data[0][2] = inv_det * G;
            out.data[1][2] = inv_det * H;
            out.data[2][2] = inv_det * I;

            return out;
        }

        pub fn atomicAdd(mat_1: *@This(), mat_2: @This()) void {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    _ = @atomicRmw(DataType, &mat_1.data[i][j], .Add, mat_2.data[i][j], .acq_rel);
                }
            }
        }

        pub fn atomicSub(mat_1: *@This(), mat_2: @This()) void {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    _ = @atomicRmw(DataType, &mat_1.data[i][j], .Sub, mat_2.data[i][j], .acq_rel);
                }
            }
        }

        pub fn atomicMinElem(mat_1: *@This(), mat_2: @This()) void {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    _ = @atomicRmw(DataType, &mat_1.data[i][j], .Min, mat_2.data[i][j], .acq_rel);
                }
            }
        }

        pub fn atomicMaxElem(mat_1: *@This(), mat_2: @This()) void {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    _ = @atomicRmw(DataType, &mat_1.data[i][j], .Max, mat_2.data[i][j], .acq_rel);
                }
            }
        }
    };
}

pub fn LargeVector(comptime DataType: type) type {
    return struct {
        data: std.ArrayList(DataType),
        len: usize = 0,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) @This() {
            const data = std.ArrayList(DataType).init(allocator);
            return .{ .data = data, .allocator = allocator };
        }

        pub fn deinit(self: *@This()) void {
            self.data.deinit();
        }

        pub fn append(self: *@This(), value: DataType) !void {
            self.len += 1;
            try self.data.append(value);
        }

        pub fn insert(self: *@This(), i: usize, value: DataType) !void {
            self.len += 1;
            try self.data.insert(i, value);
        }

        pub fn appendNTimes(self: *@This(), value: DataType, n: usize) !void {
            self.len += n;
            try self.data.appendNTimes(value, n);
        }

        pub fn setAllZeros(self: *@This()) !void {
            for (0..self.len) |i| {
                self.data.items[i] = 0;
            }
        }

        pub fn get(self: *@This(), i: usize) DataType {
            return self.data.items[i];
        }

        pub fn set(self: *@This(), i: usize, val: DataType) void {
            self.data.items[i] = val;
        }

        pub fn l2Norm(self: *@This()) DataType {
            var out: DataType = 0;
            for (0..self.len) |i| {
                out += std.math.pow(DataType, self.data.items[i], 2);
            }
            return @sqrt(out);
        }

        pub fn add(vec_1: *@This(), vec_2: *@This(), result_vec: *@This()) void {
            for (0..result_vec.len) |i| {
                result_vec.data.items[i] = vec_1.data.items[i] + vec_2.data.items[i];
            }
        }

        pub fn sub(vec_1: *@This(), vec_2: *@This(), result_vec: *@This()) void {
            for (0..result_vec.len) |i| {
                result_vec.data.items[i] = vec_1.data.items[i] - vec_2.data.items[i];
            }
        }

        pub fn mulElem(vec_1: *@This(), vec_2: *@This(), result_vec: *@This()) void {
            for (0..result_vec.len) |i| {
                result_vec.data.items[i] = vec_1.data.items[i] * vec_2.data.items[i];
            }
        }

        pub fn divElem(vec_1: *@This(), vec_2: *@This(), result_vec: *@This()) void {
            for (0..result_vec.len) |i| {
                result_vec.data.items[i] = vec_1.data.items[i] / vec_2.data.items[i];
            }
        }

        pub fn dot(vec_1: *@This(), vec_2: *@This()) DataType {
            var out = 0;
            for (0..vec_1.len) |i| {
                out += vec_1.data.items[i] * vec_2.data.items[i];
            }
        }

        pub fn scale(vec: *@This(), scalar: DataType, result_vec: *@This()) void {
            for (0..result_vec.len) |i| {
                result_vec.data.items[i] = vec.data.items[i] * scalar;
            }
        }

        pub fn clone(vec: *@This(), result_vec: *@This()) void {
            for (0..result_vec.len) |i| {
                result_vec.data.items[i] = vec.data.items[i];
            }
        }
    };
}

pub fn SparseMatrixCSR(comptime DataType: type) type {
    return struct {
        data: LargeVector(DataType),
        rows: LargeVector(usize),
        cols: LargeVector(usize),
        n_rows: usize = 0,
        n_cols: usize = 0,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) @This() {
            const data = LargeVector(DataType).init(allocator);
            const rows = LargeVector(usize).init(allocator);
            const cols = LargeVector(usize).init(allocator);

            return .{
                .data = data,
                .rows = rows,
                .cols = cols,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.data.deinit();
            self.rows.deinit();
            self.cols.deinit();
        }

        pub fn setSize(self: *@This(), n_rows: usize, n_cols: usize) !void {
            self.n_rows = n_rows;
            self.n_cols = n_cols;
            try self.rows.appendNTimes(0, n_rows + 1);
        }

        pub fn get(self: *@This(), row: usize, col: usize) DataType {
            const start_row_i = self.rows.get(row);
            const end_row_i = self.rows.get(row + 1);

            var val: DataType = 0;

            for (start_row_i..end_row_i) |i| {
                if (self.cols.get(i) == col) {
                    val = self.data.get(i);
                }
            }

            return val;
        }

        pub fn set(self: *@This(), row: usize, col: usize, val: DataType) !void {
            var has_entry: bool = false;
            var index: usize = 0;

            const start_row_i = self.rows.get(row);
            const end_row_i = self.rows.get(row + 1);

            for (start_row_i..end_row_i) |i| {
                if (self.cols.get(i) == col) {
                    index = i;
                    has_entry = true;
                }
            }

            if (val != 0) {
                if (has_entry) {
                    self.data.set(index, val);
                } else {
                    try self.data.insert(end_row_i, val);
                    try self.cols.insert(end_row_i, col);

                    for (row + 1..self.n_rows + 1) |i| {
                        // self.rows.set(i, self.rows.get(i) + 1)
                        self.rows.data.items[i] += 1;
                    }
                }
            }
        }

        pub fn scale(self: *@This(), scalar: DataType, result_mat: *@This()) void {
            for (0..self.data.len) |i| {
                result_mat.data.items[i] = scalar * self.data.items[i];
            }
        }

        pub fn mulVec(self: *@This(), vec: *LargeVector(DataType), result_vec: *LargeVector(DataType)) void {
            for (0..self.n_rows) |i| {
                for (self.rows.get(i)..self.rows.get(i + 1)) |col_i| {
                    const j = self.cols.get(col_i);
                    result_vec.data.items[i] += vec.data.items[j] * self.data.data.items[col_i];
                }
            }
        }
    };
}
