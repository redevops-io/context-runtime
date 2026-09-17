# Mojo 1.1 port of context_runtime/geospatial/engine.py — a FAITHFUL mirror of the reference
# algorithm and its epsilon/tie-break decisions (1e-9 on-edge, 1e-12 collinearity). Do not "improve".
# Rings are passed as a flat interleaved List[Float64] [x0,y0,x1,y1,...] plus a vertex count.

def _sign(v: Float64) -> Int:
    var pos = 1 if v > 1e-12 else 0
    var neg = 1 if v < -1e-12 else 0
    return pos - neg

def point_in_ring(px: Float64, py: Float64, r: List[Float64], n: Int) -> Bool:
    var inside = False
    for i in range(n):
        var j = (i + 1) % n
        var x1 = r[2 * i];     var y1 = r[2 * i + 1]
        var x2 = r[2 * j];     var y2 = r[2 * j + 1]
        # on-edge tie-break: on a boundary counts as inside (mirrors the reference exactly)
        if min(y1, y2) <= py and py <= max(y1, y2) and min(x1, x2) - 1e-12 <= px and px <= max(x1, x2) + 1e-12:
            if abs((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)) < 1e-9:
                return True
        if (y1 > py) != (y2 > py):
            var xint = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < xint:
                inside = not inside
    return inside

def signed_area(r: List[Float64], n: Int) -> Float64:
    var s = 0.0
    for i in range(n):
        var j = (i + 1) % n
        s += r[2 * i] * r[2 * j + 1] - r[2 * j] * r[2 * i + 1]
    return s / 2.0

def centroid_x(r: List[Float64], n: Int) -> Float64:
    var a = signed_area(r, n)
    if abs(a) < 1e-12:
        var sx = 0.0
        for i in range(n):
            sx += r[2 * i]
        return sx / Float64(n if n > 0 else 1)
    var cx = 0.0
    for i in range(n):
        var j = (i + 1) % n
        var cross = r[2 * i] * r[2 * j + 1] - r[2 * j] * r[2 * i + 1]
        cx += (r[2 * i] + r[2 * j]) * cross
    return cx / (6.0 * a)

def centroid_y(r: List[Float64], n: Int) -> Float64:
    var a = signed_area(r, n)
    if abs(a) < 1e-12:
        var sy = 0.0
        for i in range(n):
            sy += r[2 * i + 1]
        return sy / Float64(n if n > 0 else 1)
    var cy = 0.0
    for i in range(n):
        var j = (i + 1) % n
        var cross = r[2 * i] * r[2 * j + 1] - r[2 * j] * r[2 * i + 1]
        cy += (r[2 * i + 1] + r[2 * j + 1]) * cross
    return cy / (6.0 * a)

def _seg_cross(p1x: Float64, p1y: Float64, p2x: Float64, p2y: Float64,
               p3x: Float64, p3y: Float64, p4x: Float64, p4y: Float64) -> Bool:
    var d1 = _sign((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x))
    var d2 = _sign((p4x - p3x) * (p2y - p3y) - (p4y - p3y) * (p2x - p3x))
    var d3 = _sign((p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x))
    var d4 = _sign((p2x - p1x) * (p4y - p1y) - (p2y - p1y) * (p4x - p1x))
    return d1 != d2 and d3 != d4

def _bbox_intersects(ax0: Float64, ay0: Float64, ax1: Float64, ay1: Float64,
                     bx0: Float64, by0: Float64, bx1: Float64, by1: Float64) -> Bool:
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

def polygons_intersect(a: List[Float64], na: Int, b: List[Float64], nb: Int) -> Bool:
    var ax0 = a[0]; var ay0 = a[1]; var ax1 = a[0]; var ay1 = a[1]
    for i in range(na):
        ax0 = min(ax0, a[2 * i]); ax1 = max(ax1, a[2 * i])
        ay0 = min(ay0, a[2 * i + 1]); ay1 = max(ay1, a[2 * i + 1])
    var bx0 = b[0]; var by0 = b[1]; var bx1 = b[0]; var by1 = b[1]
    for i in range(nb):
        bx0 = min(bx0, b[2 * i]); bx1 = max(bx1, b[2 * i])
        by0 = min(by0, b[2 * i + 1]); by1 = max(by1, b[2 * i + 1])
    if not _bbox_intersects(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        return False
    for i in range(na):
        if point_in_ring(a[2 * i], a[2 * i + 1], b, nb):
            return True
    for i in range(nb):
        if point_in_ring(b[2 * i], b[2 * i + 1], a, na):
            return True
    for i in range(na):
        var ia = (i + 1) % na
        for j in range(nb):
            var jb = (j + 1) % nb
            if _seg_cross(a[2 * i], a[2 * i + 1], a[2 * ia], a[2 * ia + 1],
                          b[2 * j], b[2 * j + 1], b[2 * jb], b[2 * jb + 1]):
                return True
    return False
