
/* override */ const fragmentsPerPixel: u32 = 10;

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

@group(0)
@binding(4)
var myTexture: texture_2d<f32>;

fn setPixelColor(pos: vec2<u32>, color: vec4<f32>, depth: f32) {
    if (pos.x >= u32(uniforms.width) || pos.y >= u32(uniforms.height) || depth < 0. || depth > 1.) {
        return;
    }

    var index = pos.x + pos.y * u32(uniforms.width);
    
    let mappedDepth     = mix(0.5, 1.0, depth);                       // make sure depth has always the same exponent
    var depthBits: u32  = bitcast<u32>(mappedDepth);                  // extract bits
    depthBits           = depthBits << 9;                             // remove sign and exponent
    depthBits           = insertBits(depthBits, 0u, 0, 12);           // reserve last 12 bits for color

    var colorBits: u32 = pack4x8unorm(vec4<f32>(color.rgb, 0.));      // [a, b, g, r]
    var bhg: u32 = extractBits(colorBits, 12, 12);                    // blue channel and half green
    let depthBHG = depthBits | bhg;

    var hgr: u32 = extractBits(colorBits, 0, 12);                    // half green and red channel
    let depthHGR = depthBits | hgr;

    atomicMin(&outputColorBuffer[index], depthBHG);
    atomicMin(&outputColorBuffer2[index], depthHGR);
}

fn ndcToViewport(ndc: vec2<f32>) -> vec2<i32> {
    var x = (uniforms.width - 1.) * .5 * (1. + ndc.x);
    var y = (uniforms.height - 1.) * .5 * (1. + ndc.y);

    return vec2<i32>(i32(x), i32(y));
}

fn project(position: vec3<f32>) -> vec4<f32> {
    let projection = uniforms.viewProjection * uniforms.model * vec4<f32>(position, 1.0);

    // needed by screen space clipping, see drawTriangle
    if (projection.w < 0.001 ) {
        return vec4<f32>(0.0, 0.0, 0.0, projection.w);
    }

    return vec4<f32>(projection.x / projection.w, projection.y / projection.w, projection.z / projection.w, projection.w);
}

fn barycentric2d(c: vec2<i32>, p1: vec2<i32>, p2: vec2<i32>, p3: vec2<i32>) -> vec3<f32> {
    let cf = vec2<f32>(f32(c.x), f32(c.y));
    let p1f = vec2<f32>(f32(p1.x), f32(p1.y));
    let p2f = vec2<f32>(f32(p2.x), f32(p2.y));
    let p3f = vec2<f32>(f32(p3.x), f32(p3.y));

    let v0 = p2f - p1f;
    let v1 = p3f - p1f;
    let v2 = cf - p1f;
    let den = v0.x * v1.y - v1.x * v0.y;
    let v = (v2.x * v1.y - v1.x * v2.y) / den;
    let w = (v0.x * v2.y - v2.x * v0.y) / den;
    let u = 1.0 - v - w;

    return vec3<f32>(u, v, w);
}

fn drawTriangle(p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, p1UV: vec2<f32>, p2UV: vec2<f32>, p3UV: vec2<f32>, color: vec4<f32>) {
    let p1v = ndcToViewport(p1.xy);
    let p2v = ndcToViewport(p2.xy);
    let p3v = ndcToViewport(p3.xy);

    // Get the triangle bounding box performing additional screen space clipping
    let minX = max( min(p1v.x, min(p2v.x, p3v.x)), 0 );
    let maxX = min( max(p1v.x, max(p2v.x, p3v.x)), i32(uniforms.width) );
    let minY = max( min(p1v.y, min(p2v.y, p3v.y)), 0 );
    let maxY = min( max(p1v.y, max(p2v.y, p3v.y)), i32(uniforms.height) );
    let dims = textureDimensions(myTexture);

    // screen space clipping for z coords
    if (p1.z <= 0 || p2.z <= 0 || p3.z <= 0) {
        return;
    }

    for (var x = minX; x < maxX; x = x + 1) {
        for (var y = minY; y < maxY; y = y + 1) {
            let b = barycentric2d(vec2<i32>(x, y), p1v, p2v, p3v);
            // should use 1 and 0, added an epsilon to avoid small holes between polygons
            if (all(b <= vec3<f32>(1.00001, 1.00001, 1.00001)) && 
                all(b >= vec3<f32>(-0.00001, -0.00001, -0.00001))) {
                
                // perspective-correct uvs
                let oneOverW =  b.x * (1.0 / p1.w) + b.y * (1.0 / p2.w) + b.z * (1.0 / p3.w);
                let uv = (b.x * (p1UV / p1.w) + b.y * (p2UV / p2.w) + b.z * (p3UV / p3.w)) / oneOverW;
                let col = textureLoad(myTexture, vec2<i32>(uv * vec2<f32>(dims)), 0);

                // perspective-correct depth
                let depth = b.x / p1.z + b.y / p2.z + b.z / p3.z;

                setPixelColor(vec2<u32>(u32(x), u32(y)), col, 1. / depth);
            }
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(uniforms.width) || id.y >= u32(uniforms.height)) {
        return;
    }

    var index = id.x + id.y * u32(uniforms.width);
    atomicStore(&outputColorBuffer[index], 0xFFFFF000u);
    atomicStore(&outputColorBuffer2[index], 0xFFFFF000u);
}

@compute
@workgroup_size(256, 1)
fn rasterizer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(uniforms.numTris)) {
        return;
    }
    let stride = u32(uniforms.stride);
    let baseIndex = id.x * 3 * stride; // three vertices with stride data
    let v1_x = vertexBuffer[baseIndex + 0u];
    let v1_y = vertexBuffer[baseIndex + 1u];
    let v1_z = vertexBuffer[baseIndex + 2u];
    let v1 = vec3<f32>(v1_x, v1_y, v1_z);
    let v1_u = vertexBuffer[baseIndex + 3u];
    let v1_v = vertexBuffer[baseIndex + 4u];
    let v1_texCoord = vec2<f32>(v1_u, v1_v);

    let v2_x = vertexBuffer[baseIndex + stride];
    let v2_y = vertexBuffer[baseIndex + stride + 1u];
    let v2_z = vertexBuffer[baseIndex + stride + 2u];
    let v2 = vec3<f32>(v2_x, v2_y, v2_z); 
    let v2_u = vertexBuffer[baseIndex + stride + 3u];
    let v2_v = vertexBuffer[baseIndex + stride + 4u];
    let v2_texCoord = vec2<f32>(v2_u, v2_v);

    let v3_x = vertexBuffer[baseIndex + stride * 2u];
    let v3_y = vertexBuffer[baseIndex + stride * 2u + 1u];
    let v3_z = vertexBuffer[baseIndex + stride * 2u + 2u];
    let v3 = vec3<f32>(v3_x, v3_y, v3_z);  
    let v3_u = vertexBuffer[baseIndex + stride * 2u + 3u];
    let v3_v = vertexBuffer[baseIndex + stride * 2u + 4u];
    let v3_texCoord = vec2<f32>(v3_u, v3_v);

    let color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    //outputColorBuffer[index] = pack4x8unorm(color);
    // setPixelColor(project(v1), color);
    // setPixelColor(project(v2), color);
    // setPixelColor(project(v3), color);
    drawTriangle(project(v1), project(v2), project(v3),
    v1_texCoord, v2_texCoord, v3_texCoord, vec4<f32>(1.0, 1.0, 1.0, 1.0));
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