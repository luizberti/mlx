//! Build-time generator: parses mlx-c op headers and emits the mlx module root
//! (the src/root.zig template verbatim, then one ziggy wrapper per C function)
//! plus Scope.zig (the src/Scope.zig template verbatim, then one chain method
//! per Array-first single-out op) plus Array.zig (the src/Array.zig template
//! verbatim, then one fallible method per Array-first op).
//! Usage: gen <root.zig> <Scope.zig> <Array.zig> <root-template> <scope-template> <array-template> <ops.h> [more headers...]
//! Headers named ops.h emit at top level; any other <name>.h emits `pub const name = struct {...}`.
const std = @import("std");

const Kind = enum {
    out_array, // mlx_array*
    out_arrays, // mlx_vector_array*
    in_array, // const mlx_array
    in_array_opt, // const mlx_array /* may be null */
    in_arrays, // const mlx_vector_array
    stream, // const mlx_stream
    slice_i32, // const int* + size_t <name>_num
    slice_i32_opt, // const int* /* may be null */ + size_t <name>_num
    slice_i64, // const int64_t* + size_t <name>_num
    string, // const char*
    scalar_i32,
    scalar_usize,
    scalar_bool,
    scalar_f32,
    scalar_f64,
    scalar_u64,
    dtype, // mlx_dtype
    fft_norm, // mlx_fft_norm
    opt_i32, // mlx_optional_int
    opt_f32, // mlx_optional_float
    opt_dtype, // mlx_optional_dtype
};

const Param = struct { kind: Kind, name: []const u8 };

const Decl = struct {
    cname: []const u8, // full C name, e.g. mlx_linalg_qr
    zname: []const u8, // wrapper name, e.g. qr (possibly @"..." escaped)
    outs: usize, // count of leading out params
    out_kind: Kind, // kind of out params (all outs of one decl share it)
    params: []const Param, // in order, outs included, slice lengths fused away
};

const Namespace = struct {
    name: []const u8, // "ops" means top level
    decls: std.ArrayList(Decl) = .empty,
    skipped: std.ArrayList([]const u8) = .empty,
};

const keywords = [_][]const u8{
    "var",  "fn",    "test",   "error", "type",   "and",    "or",     "if",
    "else", "while", "for",    "break", "return", "switch", "defer",  "struct",
    "enum", "union", "opaque", "pub",   "const",  "export", "extern", "inline",
};

// File-scope decls of the generated module; params and fn names must not shadow them.
const reserved = [_][]const u8{
    "core",         "cffi",           "std",       "Error", "DType",      "Array",
    "Arrays",       "Stream",         "Scope",     "Norm",  "init",       "check",
    "lastError",    "metalAvailable", "res0",      "res1",  "transforms", "Closure",
    "ValueAndGrad", "eval",           "asyncEval", "vjp",   "jvp",        "valueAndGrad",
    "checkpoint",   "compile",
};

fn contains(set: []const []const u8, s: []const u8) bool {
    for (set) |k| if (std.mem.eql(u8, s, k)) return true;
    return false;
}

fn camelize(arena: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var up = false;
    for (s) |ch| {
        if (ch == '_') {
            up = true;
            continue;
        }
        try out.append(arena, if (up) std.ascii.toUpper(ch) else ch);
        up = false;
    }
    return out.items;
}

// mlx-c param names that read poorly or collide pervasively in Zig.
fn paramAlias(name: []const u8) []const u8 {
    if (std.mem.eql(u8, name, "transpose")) return "transp"; // collides with the transpose op
    if (std.mem.eql(u8, name, "type")) return "dtype"; // Zig keyword; the param is an mlx_dtype
    if (std.mem.eql(u8, name, "logits_")) return "logits"; // upstream typo in categorical_num_samples
    if (std.mem.eql(u8, name, "mean")) return "mu"; // collides with the top-level mean op
    return name;
}

// Renamed wrapper fns: mlx_var would need @"var" keyword escaping; random.key
// and random.seed shadowed the `key`/`seed` params used across the namespace
// (reseed also says what it does: mutate the global PRNG state).
fn fnAlias(zname: []const u8) []const u8 {
    if (std.mem.eql(u8, zname, "var")) return "variance";
    if (std.mem.eql(u8, zname, "key")) return "prng";
    if (std.mem.eql(u8, zname, "seed")) return "reseed";
    return zname;
}

const nullable_marker = "\x01N\x01";

fn stripComments(arena: std.mem.Allocator, text: []const u8) ![]const u8 {
    const marked = try std.mem.replaceOwned(u8, arena, text, "/* may be null */", nullable_marker);
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (std.mem.findPos(u8, marked, i, "/*")) |start| {
        try out.appendSlice(arena, marked[i..start]);
        const stop = std.mem.findPos(u8, marked, start + 2, "*/") orelse return error.UnterminatedComment;
        i = stop + 2;
    }
    try out.appendSlice(arena, marked[i..]);
    return out.items;
}

const RawParam = struct { type: []const u8, name: []const u8, nullable: bool };

fn parseRawParam(arena: std.mem.Allocator, text: []const u8) !RawParam {
    var nullable = false;
    var toks: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \t\n");
    while (it.next()) |t| {
        if (std.mem.eql(u8, t, nullable_marker)) {
            nullable = true;
            continue;
        }
        try toks.append(arena, t);
    }
    if (toks.items.len < 2) return error.BadParam;
    var name = toks.items[toks.items.len - 1];
    var ty: std.ArrayList(u8) = .empty;
    for (toks.items[0 .. toks.items.len - 1], 0..) |t, i| {
        if (i > 0) try ty.append(arena, ' ');
        try ty.appendSlice(arena, t);
    }
    if (name[0] == '*') { // "mlx_array *res" style, just in case
        try ty.append(arena, '*');
        name = name[1..];
    }
    return .{ .type = ty.items, .name = name, .nullable = nullable };
}

fn classify(raw: RawParam) ?Kind {
    const map = [_]struct { []const u8, Kind }{
        .{ "mlx_array*", .out_array },
        .{ "mlx_vector_array*", .out_arrays },
        .{ "const mlx_array", .in_array },
        .{ "const mlx_vector_array", .in_arrays },
        .{ "const mlx_stream", .stream },
        .{ "const int*", .slice_i32 },
        .{ "const int64_t*", .slice_i64 },
        .{ "const char*", .string },
        .{ "int", .scalar_i32 },
        .{ "size_t", .scalar_usize },
        .{ "bool", .scalar_bool },
        .{ "float", .scalar_f32 },
        .{ "double", .scalar_f64 },
        .{ "uint64_t", .scalar_u64 },
        .{ "mlx_dtype", .dtype },
        .{ "mlx_fft_norm", .fft_norm },
        .{ "mlx_optional_int", .opt_i32 },
        .{ "mlx_optional_float", .opt_f32 },
        .{ "mlx_optional_dtype", .opt_dtype },
    };
    for (map) |entry| {
        const ty, const kind = entry;
        if (std.mem.eql(u8, raw.type, ty)) {
            if (!raw.nullable) return kind;
            return switch (kind) {
                .in_array => .in_array_opt,
                .slice_i32 => .slice_i32_opt,
                else => null,
            };
        }
    }
    return null;
}

fn parseHeader(arena: std.mem.Allocator, ns: *Namespace, text: []const u8) !void {
    const clean = try stripComments(arena, text);
    var i: usize = 0;
    while (std.mem.findPos(u8, clean, i, "int mlx_")) |start| {
        i = start + 1;
        if (start > 0 and (std.ascii.isAlphanumeric(clean[start - 1]) or clean[start - 1] == '_')) continue;
        const open = std.mem.findScalarPos(u8, clean, start, '(') orelse continue;
        const cname = std.mem.trim(u8, clean[start + 4 .. open], " \t\n");
        if (cname.len == 0) continue;
        for (cname) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '_') continue;
        const close = std.mem.findScalarPos(u8, clean, open, ')') orelse continue;
        i = close;

        var raws: std.ArrayList(RawParam) = .empty;
        var ok = true;
        var pit = std.mem.splitScalar(u8, clean[open + 1 .. close], ',');
        while (pit.next()) |p| {
            if (std.mem.trim(u8, p, " \t\n").len == 0) continue;
            raws.append(arena, parseRawParam(arena, p) catch {
                ok = false;
                break;
            }) catch return error.OutOfMemory;
        }
        if (!ok) {
            try ns.skipped.append(arena, cname);
            continue;
        }

        // Classify and fuse <ptr> + size_t <name>_num pairs.
        var params: std.ArrayList(Param) = .empty;
        var outs: usize = 0;
        var out_kind: Kind = .out_array;
        var j: usize = 0;
        while (j < raws.items.len) : (j += 1) {
            const raw = raws.items[j];
            const kind = classify(raw) orelse {
                ok = false;
                break;
            };
            switch (kind) {
                .out_array, .out_arrays => {
                    if (outs != params.items.len or (outs > 0 and out_kind != kind)) {
                        ok = false; // out params must be leading and homogeneous
                        break;
                    }
                    outs += 1;
                    out_kind = kind;
                },
                .slice_i32, .slice_i32_opt, .slice_i64 => {
                    j += 1; // consume the size_t <name>_num companion
                    if (j >= raws.items.len) {
                        ok = false;
                        break;
                    }
                    const num = raws.items[j];
                    const want = try std.fmt.allocPrint(arena, "{s}_num", .{raw.name});
                    if (!std.mem.eql(u8, num.type, "size_t") or !std.mem.eql(u8, num.name, want)) {
                        ok = false;
                        break;
                    }
                },
                else => {},
            }
            try params.append(arena, .{ .kind = kind, .name = paramAlias(raw.name) });
        }
        if (!ok) {
            try ns.skipped.append(arena, cname);
            continue;
        }

        var stem = std.mem.cutPrefix(u8, cname, "mlx_") orelse cname;
        if (!std.mem.eql(u8, ns.name, "ops")) {
            const pfx = try std.fmt.allocPrint(arena, "{s}_", .{ns.name});
            stem = std.mem.cutPrefix(u8, stem, pfx) orelse stem;
        }
        const zname = try camelize(arena, stem);
        try ns.decls.append(arena, .{
            .cname = cname,
            .zname = fnAlias(zname),
            .outs = outs,
            .out_kind = out_kind,
            .params = params.items,
        });
    }
}

fn zigType(kind: Kind) []const u8 {
    return switch (kind) {
        .in_array => "Array",
        .in_array_opt => "?Array",
        .in_arrays => "Arrays",
        .stream => "Stream",
        .slice_i32 => "[]const i32",
        .slice_i32_opt => "?[]const i32",
        .slice_i64 => "[]const i64",
        .string => "[:0]const u8",
        .scalar_i32 => "i32",
        .scalar_usize => "usize",
        .scalar_bool => "bool",
        .scalar_f32 => "f32",
        .scalar_f64 => "f64",
        .scalar_u64 => "u64",
        .dtype => "DType",
        .fft_norm => "Norm",
        .opt_i32 => "?i32",
        .opt_f32 => "?f32",
        .opt_dtype => "?DType",
        .out_array, .out_arrays => unreachable,
    };
}

fn writeArg(w: *std.Io.Writer, kind: Kind, name: []const u8) !void {
    switch (kind) {
        .in_array => try w.print("{s}.info.handle", .{name}),
        .in_arrays, .stream => try w.print("{s}.h", .{name}),
        .in_array_opt => try w.print("if ({s}) |{s}_v| {s}_v.info.handle else .{{ .ctx = null }}", .{ name, name, name }),
        .slice_i32, .slice_i64 => try w.print("{s}.ptr, {s}.len", .{ name, name }),
        .slice_i32_opt => try w.print("if ({s}) |{s}_v| {s}_v.ptr else null, if ({s}) |{s}_v| {s}_v.len else 0", .{ name, name, name, name, name, name }),
        .string => try w.print("{s}.ptr", .{name}),
        .scalar_i32, .scalar_usize, .scalar_bool, .scalar_f32, .scalar_f64, .scalar_u64 => try w.print("{s}", .{name}),
        .dtype, .fft_norm => try w.print("@intFromEnum({s})", .{name}),
        .opt_i32, .opt_f32 => try w.print(".{{ .value = {s} orelse 0, .has_value = {s} != null }}", .{ name, name }),
        .opt_dtype => try w.print(".{{ .value = if ({s}) |{s}_v| @intFromEnum({s}_v) else 0, .has_value = {s} != null }}", .{ name, name, name, name }),
        .out_array, .out_arrays => unreachable,
    }
}

fn emitNamespace(arena: std.mem.Allocator, w: *std.Io.Writer, ns: *const Namespace, outer: []const []const u8) !void {
    // Names a param may not reuse: file-scope decls (incl. top-level ops, which remain
    // in scope inside nested namespaces) and this namespace's own wrappers.
    var taken: std.ArrayList([]const u8) = .empty;
    try taken.appendSlice(arena, &reserved);
    try taken.appendSlice(arena, outer);
    for (ns.decls.items) |d| try taken.append(arena, d.zname);

    for (ns.decls.items) |d| {
        const fname = if (contains(&keywords, d.zname))
            try std.fmt.allocPrint(arena, "@\"{s}\"", .{d.zname})
        else
            d.zname;

        try w.print("pub fn {s}(", .{fname});
        var names = try arena.alloc([]const u8, d.params.len);
        var first = true;
        for (d.params, 0..) |p, idx| {
            switch (p.kind) {
                .out_array, .out_arrays => continue,
                else => {},
            }
            names[idx] = if (contains(taken.items, p.name) or contains(&keywords, p.name))
                try std.fmt.allocPrint(arena, "{s}_", .{p.name})
            else
                p.name;
            if (!first) try w.writeAll(", ");
            try w.print("{s}: {s}", .{ names[idx], zigType(p.kind) });
            first = false;
        }
        const ret = if (d.outs == 0)
            "void"
        else if (d.outs == 1)
            (if (d.out_kind == .out_array) "Array" else "Arrays")
        else
            "struct { Array, Array }";
        try w.print(") Error!{s} {{\n", .{ret});

        const ctor = if (d.out_kind == .out_array) "mlx_array_new" else "mlx_vector_array_new";
        for (0..d.outs) |o| try w.print("    var res{d} = cffi.{s}();\n", .{ o, ctor });

        try w.print("    try check(cffi.{s}(", .{d.cname});
        var o: usize = 0;
        for (d.params, 0..) |p, idx| {
            if (idx > 0) try w.writeAll(", ");
            switch (p.kind) {
                .out_array, .out_arrays => {
                    try w.print("&res{d}", .{o});
                    o += 1;
                },
                else => try writeArg(w, p.kind, names[idx]),
            }
        }
        try w.writeAll("));\n");

        if (d.outs == 1) {
            if (d.out_kind == .out_array) {
                try w.writeAll("    return .{ .info = .{ .handle = res0 } };\n");
            } else {
                try w.writeAll("    return .{ .h = res0 };\n");
            }
        } else if (d.outs > 1) {
            try w.writeAll("    return .{");
            for (0..d.outs) |k| try w.print("{s} .{{ .info = .{{ .handle = res{d} }} }}", .{ if (k > 0) "," else "", k });
            try w.writeAll(" };\n");
        }
        try w.writeAll("}\n");
    }

    if (ns.skipped.items.len > 0) {
        try w.writeAll("// Not wrapped (unrecognized signature) — use the raw `c` import:\n");
        for (ns.skipped.items) |s| try w.print("//   {s}\n", .{s});
    }
}

// Chain methods on Scope: each eligible op threads self.result through and
// no-ops once it holds an error. Eligible = single mlx_array out, first input
// is a required mlx_array (becomes self.result), last param is the stream
// (becomes self.stream). Everything else keeps the free-function signature.
fn emitScopeMethods(arena: std.mem.Allocator, w: *std.Io.Writer, ns: *const Namespace) !void {
    // Decls of the Scope.zig template; method and param names must not collide.
    const scope_reserved = [_][]const u8{
        "stdz",   "core",       "cffi",  "Error", "Array",  "Arrays", "DType",
        "Stream", "ChainError", "root",  "init",  "deinit", "enter",  "collect",
        "track",  "escape",     "alloc", "arena", "stream", "result", "unwrapped",
        "self",
    };
    var taken: std.ArrayList([]const u8) = .empty;
    try taken.appendSlice(arena, &scope_reserved);
    for (ns.decls.items) |d| try taken.append(arena, d.zname);

    for (ns.decls.items) |d| {
        if (d.outs != 1 or d.out_kind != .out_array) continue;
        if (d.params.len < 3) continue; // out + leading array + stream
        if (d.params[1].kind != .in_array) continue;
        if (d.params[d.params.len - 1].kind != .stream) continue;
        if (contains(&scope_reserved, d.zname)) {
            try w.print("// Not a chain method (collides with a Scope decl) — use the free function: {s}\n", .{d.cname});
            continue;
        }
        const fname = if (contains(&keywords, d.zname))
            try std.fmt.allocPrint(arena, "@\"{s}\"", .{d.zname})
        else
            d.zname;

        try w.print("pub fn {s}(self: *@This()", .{fname});
        const mids = d.params[2 .. d.params.len - 1];
        var names = try arena.alloc([]const u8, mids.len);
        for (mids, 0..) |p, idx| {
            names[idx] = if (contains(taken.items, p.name) or contains(&keywords, p.name))
                try std.fmt.allocPrint(arena, "{s}_", .{p.name})
            else
                p.name;
            try w.print(", {s}: {s}", .{ names[idx], zigType(p.kind) });
        }
        try w.writeAll(") *@This() {\n");
        try w.writeAll("    const unwrapped = self.result catch return self;\n");
        try w.print("    self.result = self.track(root.{s}(unwrapped", .{fname});
        for (names) |n| try w.print(", {s}", .{n});
        try w.writeAll(", self.stream));\n    return self;\n}\n");
    }
}

// Ops as methods on Array: thin forwarders to the free functions for any op
// whose first input is a required mlx_array and whose last param is the stream.
// Unlike Scope chains these are fallible per call, and multi-out ops are fine.
fn emitArrayMethods(arena: std.mem.Allocator, w: *std.Io.Writer, ns: *const Namespace) !void {
    // Decls of the Array.zig template; method and param names must not collide.
    const array_reserved = [_][]const u8{
        "stdz",   "core",   "root",   "cffi",   "check",  "Error",     "Arrays",
        "DType",  "Stream", "itemFn", "dataFn", "scalar", "fromSlice", "deinit",
        "assign", "clone",  "info",   "Info",   "eval",   "item",      "data",
        "format", "self",
    };
    var taken: std.ArrayList([]const u8) = .empty;
    try taken.appendSlice(arena, &array_reserved);
    for (ns.decls.items) |d| try taken.append(arena, d.zname);

    for (ns.decls.items) |d| {
        if (d.outs == 0) continue;
        if (d.params.len < d.outs + 2) continue; // at least self + stream
        if (d.params[d.outs].kind != .in_array) continue;
        if (d.params[d.params.len - 1].kind != .stream) continue;
        if (contains(&array_reserved, d.zname)) {
            try w.print("// Not a method (collides with an Array decl) — use the free function: {s}\n", .{d.cname});
            continue;
        }
        const fname = if (contains(&keywords, d.zname))
            try std.fmt.allocPrint(arena, "@\"{s}\"", .{d.zname})
        else
            d.zname;

        try w.print("pub fn {s}(self: @This()", .{fname});
        const rest = d.params[d.outs + 1 ..];
        var names = try arena.alloc([]const u8, rest.len);
        for (rest, 0..) |p, idx| {
            names[idx] = if (contains(taken.items, p.name) or contains(&keywords, p.name))
                try std.fmt.allocPrint(arena, "{s}_", .{p.name})
            else
                p.name;
            const ty = switch (p.kind) {
                .in_array => "@This()",
                .in_array_opt => "?@This()",
                else => zigType(p.kind),
            };
            try w.print(", {s}: {s}", .{ names[idx], ty });
        }
        const ret = if (d.outs == 1)
            (if (d.out_kind == .out_array) "@This()" else "Arrays")
        else
            "struct { @This(), @This() }";
        try w.print(") Error!{s} {{\n", .{ret});
        try w.print("    return root.{s}(self", .{fname});
        for (names) |n| try w.print(", {s}", .{n});
        try w.writeAll(");\n}\n");
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 8) return error.Usage; // gen <root.zig> <Scope.zig> <Array.zig> <root-template> <scope-template> <array-template> <header...>

    var namespaces: std.ArrayList(Namespace) = .empty;
    for (args[7..]) |path| {
        const base = std.fs.path.basename(path);
        const name = std.mem.cutSuffix(u8, base, ".h") orelse base;
        var ns: Namespace = .{ .name = name };
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(8 << 20));
        try parseHeader(arena, &ns, text);
        try namespaces.append(arena, ns);
    }

    // Everything declared at the top level of the generated file stays in scope
    // inside nested namespaces, so params anywhere must not shadow any of it.
    var top: std.ArrayList([]const u8) = .empty;
    for (namespaces.items) |ns| {
        if (std.mem.eql(u8, ns.name, "ops")) {
            for (ns.decls.items) |d| try top.append(arena, d.zname);
        } else {
            try top.append(arena, ns.name);
        }
    }

    var root_file = try std.Io.Dir.cwd().createFile(io, args[1], .{});
    defer root_file.close(io);
    var rbuf: [64 * 1024]u8 = undefined;
    var rfw = root_file.writer(io, &rbuf);
    const w = &rfw.interface;

    try w.writeAll(try std.Io.Dir.cwd().readFileAlloc(io, args[4], arena, .limited(1 << 20)));
    try w.writeAll(
        \\
        \\// Everything below is generated by gen.zig from the mlx-c headers. Do not edit.
        \\pub const Norm = enum(c_uint) { backward = 0, ortho = 1, forward = 2 };
        \\
        \\
    );
    for (namespaces.items) |*ns| {
        if (std.mem.eql(u8, ns.name, "ops")) {
            try emitNamespace(arena, w, ns, top.items);
        } else {
            try w.print("pub const {s} = struct {{\n", .{ns.name});
            try emitNamespace(arena, w, ns, top.items);
            try w.writeAll("};\n");
        }
    }
    try w.flush();

    var scope_file = try std.Io.Dir.cwd().createFile(io, args[2], .{});
    defer scope_file.close(io);
    var sbuf: [64 * 1024]u8 = undefined;
    var sfw = scope_file.writer(io, &sbuf);
    const sw = &sfw.interface;

    try sw.writeAll(try std.Io.Dir.cwd().readFileAlloc(io, args[5], arena, .limited(1 << 20)));
    try sw.writeAll(
        \\
        \\// Everything below is generated by gen.zig from the mlx-c headers. Do not edit.
        \\
        \\
    );
    for (namespaces.items) |*ns| {
        if (std.mem.eql(u8, ns.name, "ops")) try emitScopeMethods(arena, sw, ns);
    }
    try sw.flush();

    var array_file = try std.Io.Dir.cwd().createFile(io, args[3], .{});
    defer array_file.close(io);
    var abuf: [64 * 1024]u8 = undefined;
    var afw = array_file.writer(io, &abuf);
    const aw = &afw.interface;

    try aw.writeAll(try std.Io.Dir.cwd().readFileAlloc(io, args[6], arena, .limited(1 << 20)));
    try aw.writeAll(
        \\
        \\// Everything below is generated by gen.zig from the mlx-c headers. Do not edit.
        \\
        \\
    );
    for (namespaces.items) |*ns| {
        if (std.mem.eql(u8, ns.name, "ops")) try emitArrayMethods(arena, aw, ns);
    }
    try aw.flush();
}
