

struct colorBuffer {
    values: array<u32>,
};

struct UBO {
    viewProjection: mat4x4<f32>,
    model: mat4x4<f32>,
    width: f32,
    height: f32,
    numTris: f32,
    stride: f32,
};

@group(0)
@binding(0)
var<storage, read_write> outputColorBuffer: array<atomic<u32>>;

@group(0)
@binding(1)
var<storage, read_write> outputColorBuffer2: array<atomic<u32>>;

@group(0)
@binding(2)
var<uniform> uniforms: UBO;

@group(0)
@binding(3)
var<storage, read> vertexBuffer: array<f32>;

// @group(1)
// @binding(2)
// var<storage, read_write> depthBuffer: array<atomic<u32>>;

fn setPixelColor(pos: vec2<u32>, color: vec4<f32>, depth: f32) {
    if (pos.x >= u32(uniforms.width) || pos.y >= u32(uniforms.height)) {
        return;
    }
    let mappedDepth = mix(0.5, 1.0, depth);                           // make sure depth has always the same exponent
    var depthBits: u32  = bitcast<u32>(mappedDepth);                  // extract bits
    depthBits           = depthBits << 9;                             // remove sign and exponent
    depthBits           = insertBits(depthBits, 0u, 0, 12);           // reserve last 12 bits for color
                                                                     
                                                                      // 32 ...... 0  bit index
    var colorBits: u32 = pack4x8unorm(vec4<f32>(color.rgb, 0.));      // [a, b, g, r]
    var bhg: u32 = extractBits(colorBits, 12, 12);                    // blue channel and half green
    let depthBHG = depthBits | bhg;

    var hgr: u32 = extractBits(colorBits, 0, 12);                    // half green and red channel
    let depthHGR = depthBits | hgr;

    atomicMin(&outputColorBuffer[pos.x + pos.y * u32(uniforms.width)], depthBHG);
    atomicMin(&outputColorBuffer2[pos.x + pos.y * u32(uniforms.width)], depthHGR);
}

fn ndcToViewport(ndc: vec2<f32>) -> vec2<u32> {
    var x = (uniforms.width - 1.) * .5 * (1. + ndc.x);
    var y = (uniforms.height - 1.) * .5 * (1. + ndc.y);

    // TODO should clip here
    x = clamp(x, 0., uniforms.width - 1);
    y = clamp(y, 0., uniforms.height - 1);

    return vec2<u32>(u32(x), u32(y));
}

fn project(position: vec3<f32>) -> vec3<f32> {
    let projection = uniforms.viewProjection * uniforms.model * vec4<f32>(position, 1.0);

    return vec3<f32>(projection.x / projection.w, projection.y / projection.w, projection.z / projection.w);
}

fn isign(i: i32) -> i32 {
    if (i >= 0) {
        return 1;
    }
    else {
        return -1;
    }
}

fn barycentric2d(c: vec2<u32>, p1: vec2<u32>, p2: vec2<u32>, p3: vec2<u32>) -> vec3<f32> {
    let cf = vec2<f32>(f32(c.x), f32(c.y));
    let p1f = vec2<f32>(f32(p1.x), f32(p1.y));
    let p2f = vec2<f32>(f32(p2.x), f32(p2.y));
    let p3f = vec2<f32>(f32(p3.x), f32(p3.y));

    let v0 = p2f - p1f;
    let v1 = p3f - p1f;
    let v2 = cf - p1f;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return vec3<f32>(v, w, u);
}

fn drawTriangle(p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, color: vec4<f32>) {
    let p1v = ndcToViewport(p1.xy);
    let p2v = ndcToViewport(p2.xy);
    let p3v = ndcToViewport(p3.xy);
    let minX = min(p1v.x, min(p2v.x, p3v.x));
    let maxX = max(p1v.x, max(p2v.x, p3v.x));
    let minY = min(p1v.y, min(p2v.y, p3v.y));
    let maxY = max(p1v.y, max(p2v.y, p3v.y));

    for (var x = minX; x < maxX; x = x + 1) {
        for (var y = minY; y < maxY; y = y + 1) {
            let b = barycentric2d(vec2<u32>(x, y), p1v, p2v, p3v);
            if (all(b <= vec3<f32>(1.0, 1.0, 1.0)) && 
                all(b >= vec3<f32>(0.0, 0.0, 0.0))) {
                let depth = b.x / p1.z + b.y / p2.z + b.z / p3.z; 
                setPixelColor(vec2<u32>(x, y), vec4<f32>(b, 1.0), 1. / depth);
            }
        }
    }
}

fn drawLineBresenham(p1: vec2<u32>, p2: vec2<u32>, color: vec4<f32>) {
    var dx = i32(p2.x - p1.x);
    var dy = i32(p2.y - p1.y);
    var absdx = abs(dx);
    var absdy = abs(dy);

    var x = i32(p1.x);
    var y = i32(p1.y);

    let xi = isign(dx);
    let yi = isign(dy);

    setPixelColor(vec2<u32>(u32(x), u32(y)), color, 0.5);

    // slope < 1
    if (absdx > absdy) {
        // d = f(x_0 + 1, y_0 + .5) - f(x_0, y_0)
        var d = 2 * absdy - absdx;

        for (var i = 0; i < absdx; i++) {
            x = x + xi;
            if (d < 0) {
                d = d + 2 * absdy;
            } else {
                y = y + yi;
                d = d + 2 * (absdy - absdx); 
            }
            setPixelColor(vec2<u32>(u32(x), u32(y)), color, 0.5);
        }
    } else { // slope >= 1, switch x and y
        var d = 2 * absdx - absdy;

        for(var i = 0; i < absdy ; i++) {
            y = y + yi;
            if (d < 0) {
                d = d + 2 * absdx;
            } else {
                x = x + xi;
                d = d + 2 * (absdx - absdy);
            }
            setPixelColor(vec2<u32>(u32(x), u32(y)), color, 0.5);
        }
    }
}

fn drawLineSimple(p1: vec2<u32>, p2: vec2<u32>, color: vec4<f32>) {
    let p1f = vec2<f32>(p1);
    let p2f = vec2<f32>(p2);
    let dist = ceil(distance(p1f, p2f));
    for (var i = 0.; i < dist; i = i + 1.) {
        let pixelPos = mix(p1f, p2f, i / dist);
        setPixelColor(vec2<u32>(pixelPos), color, 0.5);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(uniforms.width) || id.y >= u32(uniforms.height)) {
        return;
    }

    atomicStore(&outputColorBuffer[id.x + id.y * u32(uniforms.width)],  0xFFFFF000u);
    atomicStore(&outputColorBuffer2[id.x + id.y * u32(uniforms.width)], 0xFFFFF000u);
    //setPixelColor(id.xy, vec4(1., 1., 1., 1.), 0.);
}

@compute
@workgroup_size(256, 1)
fn rasterizer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(uniforms.numTris)) {
        return;
    }
    let baseIndex = id.x * 3 * u32(uniforms.stride); // three vertices with stride data
    let v1_x = vertexBuffer[baseIndex + 0u];
    let v1_y = vertexBuffer[baseIndex + 1u];
    let v1_z = vertexBuffer[baseIndex + 2u];
    let v1 = vec3<f32>(v1_x, v1_y, v1_z);

    let v2_x = vertexBuffer[baseIndex + 3u];
    let v2_y = vertexBuffer[baseIndex + 4u];
    let v2_z = vertexBuffer[baseIndex + 5u];
    let v2 = vec3<f32>(v2_x, v2_y, v2_z); 

    let v3_x = vertexBuffer[baseIndex + 6u];
    let v3_y = vertexBuffer[baseIndex + 7u];
    let v3_z = vertexBuffer[baseIndex + 8u];
    let v3 = vec3<f32>(v3_x, v3_y, v3_z);  

    let color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    //outputColorBuffer[index] = pack4x8unorm(color);
    // setPixelColor(project(v1), color);
    // setPixelColor(project(v2), color);
    // setPixelColor(project(v3), color);
    drawTriangle(project(v1), project(v2), project(v3), vec4<f32>(1.0, 1.0, 0.0, 1.0));
    if (bool(0)) {
        // drawLineSimple(project(v1), project(v2), color);
        // drawLineSimple(project(v3), project(v2), color);
        // drawLineSimple(project(v1), project(v3), color);
    } else {
        // drawLineBresenham(project(v1), project(v2), color);
        // drawLineBresenham(project(v2), project(v3), color);
        // drawLineBresenham(project(v1), project(v3), color);
    }
}


@group(0)
@binding(1)
var<storage> finalColorBufferFS: array<u32>;

@group(0)
@binding(2)
var<storage> finalColorBuffer2FS: array<u32>;

@group(0)
@binding(0)
var<uniform> uniformsFS: UBO;

@vertex
fn fullScreenVert(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
      vec2<f32>( 1.0,  1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0,  1.0));
    
    return vec4<f32>(pos[index], 0.0, 1.0);
}

@fragment
fn fullScreenFrag(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let X = floor(coord.x);
    let Y = floor(coord.y);
    let index = u32(X + Y * uniformsFS.width);
    var depthBHG: u32 = finalColorBufferFS[index];     // decode blue and half green channel (with 20 bits of depth in front)
    var bhg: u32      = extractBits(depthBHG, 0, 12);  // remove 
    bhg = bhg << 12;                                   // save space for rhg
    
    var depthHGR: u32 = finalColorBuffer2FS[index];    // decode half green and red channel (with 20 bits of depth in front) 
    var hgr: u32      = extractBits(depthHGR, 0, 12);  // remove depth 

    var abgr: u32 = bhg | hgr;
    //insertBits(abgr, 1u, 0, 8);                          // set alpha to 1, useless in this case

    let finalColor = unpack4x8unorm(abgr);
    return vec4<f32>(finalColor.xyz, 1.0);
}