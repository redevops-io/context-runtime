# Native Mojo runner. argv: <workload> <input-file>. Reads a flat whitespace-separated float protocol,
# runs the ported geometry (timing ONLY the kernel loop = T1 native), prints results + "KERNEL_NS <n>".
# I/O + parse are intentionally untimed; the Python bridge measures the integration boundary itself.
from std.time import perf_counter_ns
from std.sys import argv
from geospatial import point_in_ring, polygons_intersect, centroid_x, centroid_y

def read_tokens(path: String) raises -> List[Float64]:
    var f = open(path, "r")
    var txt = f.read()
    f.close()
    var parts = txt.split()
    var out = List[Float64]()
    for i in range(len(parts)):
        out.append(Float64(parts[i]))
    return out^

def main() raises:
    var args = argv()
    var workload = String(args[1])
    var path = String(args[2])
    var t = read_tokens(path)
    var pos = 0
    var results = String("")

    if workload == "pip":
        var count = Int(t[pos]); pos += 1
        var flags = List[Int]()
        var t0 = perf_counter_ns()
        for _ in range(count):
            var nv = Int(t[pos]); pos += 1
            var ring = List[Float64]()
            for k in range(2 * nv):
                ring.append(t[pos + k])
            pos += 2 * nv
            var px = t[pos]; var py = t[pos + 1]; pos += 2
            flags.append(1 if point_in_ring(px, py, ring, nv) else 0)
        var t1 = perf_counter_ns()
        for i in range(len(flags)):
            results += String(flags[i]) + "\n"
        print(results, end="")
        print("KERNEL_NS", t1 - t0)

    elif workload == "pairs":
        var count = Int(t[pos]); pos += 1
        var flags = List[Int]()
        var t0 = perf_counter_ns()
        for _ in range(count):
            var na = Int(t[pos]); pos += 1
            var a = List[Float64]()
            for k in range(2 * na):
                a.append(t[pos + k])
            pos += 2 * na
            var nb = Int(t[pos]); pos += 1
            var b = List[Float64]()
            for k in range(2 * nb):
                b.append(t[pos + k])
            pos += 2 * nb
            flags.append(1 if polygons_intersect(a, na, b, nb) else 0)
        var t1 = perf_counter_ns()
        for i in range(len(flags)):
            results += String(flags[i]) + "\n"
        print(results, end="")
        print("KERNEL_NS", t1 - t0)

    elif workload == "join":
        var nl = Int(t[pos]); pos += 1
        var nr = Int(t[pos]); pos += 1
        # read zones (flat rings + counts)
        var zrings = List[List[Float64]]()
        var zn = List[Int]()
        for _ in range(nr):
            var nzv = Int(t[pos]); pos += 1
            var zr = List[Float64]()
            for k in range(2 * nzv):
                zr.append(t[pos + k])
            pos += 2 * nzv
            zrings.append(zr^); zn.append(nzv)
        # read parcels
        var prings = List[List[Float64]]()
        var pn = List[Int]()
        for _ in range(nl):
            var npv = Int(t[pos]); pos += 1
            var pr = List[Float64]()
            for k in range(2 * npv):
                pr.append(t[pos + k])
            pos += 2 * npv
            prings.append(pr^); pn.append(npv)
        # kernel: centroid each parcel, test against every zone (the O(nl*nr) path)
        var hit = List[String]()
        var t0 = perf_counter_ns()
        for pi in range(nl):
            var cx = centroid_x(prings[pi], pn[pi])
            var cy = centroid_y(prings[pi], pn[pi])
            var line = String("")
            var kcount = 0
            for zi in range(nr):
                if point_in_ring(cx, cy, zrings[zi], zn[zi]):
                    line += " " + String(zi)
                    kcount += 1
            hit.append(String(kcount) + line)
        var t1 = perf_counter_ns()
        for i in range(len(hit)):
            results += hit[i] + "\n"
        print(results, end="")
        print("KERNEL_NS", t1 - t0)
    else:
        print("unknown workload", workload)
