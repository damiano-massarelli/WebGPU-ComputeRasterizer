
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
var<storage, read_write> outputColorBuffer: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> outputDepthBuffer: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> fragmentsCounter: array<atomic<u32>>;

@group(0)
@binding(3)
var<uniform> uniforms: UBO;

@group(0)
@binding(4)
var<storage, read> vertexBuffer: array<f32>;

@group(0)
@binding(5)
var myTexture: texture_2d<f32>;

fn setPixelColor(pos: vec2<u32>, color: vec4<f32>, depth: f32) {
    if (pos.x >= u32(uniforms.width) || pos.y >= u32(uniforms.height) || depth < 0. || depth > 1.) {
        return;
    }

    var index = pos.x + pos.y * u32(uniforms.width);
    let fragmentLocalIndex = atomicAdd(&fragmentsCounter[index], 1u);
    if (fragmentLocalIndex >= fragmentsPerPixel) {
        return;
    }

    let colorDepthIndex = pos.x * fragmentsPerPixel + pos.y * u32(uniforms.width) * fragmentsPerPixel + fragmentLocalIndex;
    outputColorBuffer[colorDepthIndex] = pack4x8unorm(color);
    outputDepthBuffer[colorDepthIndex] = u32(f32(0xFFFFFFFFu) * depth);
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
    atomicStore(&fragmentsCounter[index], 0u);
    
    for (var i = 0u; i < fragmentsPerPixel; i = i + 1u) {
        let colorDepthIndex = id.x * fragmentsPerPixel + id.y * u32(uniforms.width) * fragmentsPerPixel + i;
        outputColorBuffer[colorDepthIndex] = 0xFF000000u;
        outputDepthBuffer[colorDepthIndex] = 0xFFFFFFFFu;
    }
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
var<storage> finalDepthBufferFS: array<u32>;

@group(0)
@binding(3)
var<storage> finalFragmentsCounter: array<u32>;

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
    let X = u32(coord.x);
    let Y = u32(coord.y);
    let index = X + Y * u32(uniformsFS.width);
    
    var colors:  array<u32, fragmentsPerPixel> = array<u32, fragmentsPerPixel>();
    var depths: array<u32, fragmentsPerPixel> = array<u32, fragmentsPerPixel>();

    for (var i = 0u; i < fragmentsPerPixel; i = i + 1u) {
        let colorDepthIndex = X * fragmentsPerPixel + Y * u32(uniformsFS.width) * fragmentsPerPixel + i;
        colors[i] = finalColorBufferFS[colorDepthIndex];
        depths[i] = finalDepthBufferFS[colorDepthIndex];
    }

    // Sort fragments
    //let frags = finalFragmentsCounter[index];
    for (var i = 0u; i < fragmentsPerPixel - 1; i = i + 1u) {
        var minIndex = i;
        for (var j = i + 1u; j < fragmentsPerPixel; j = j + 1u) {
            if (depths[j] > depths[minIndex]) {
                minIndex = j;
            }
        }
        let tmpColor = colors[i];
        colors[i] = colors[minIndex];
        colors[minIndex] = tmpColor;

        let tmpDepth = depths[i];
        depths[i] = depths[minIndex];
        depths[minIndex] = tmpDepth;
    }

    var finalColor: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < fragmentsPerPixel; i = i + 1u) {
        let c = unpack4x8unorm(colors[i]);
        finalColor = c.rgb * 0.7 + finalColor * 0.3;
    }

    // var minIdx = 0u;
    // var minDepth = depths[minIdx];
    // for (var i = 1u; i < fragmentsPerPixel; i = i + 1u) {
    //     if (depths[i] < minDepth) {
    //         minDepth = depths[i];
    //         minIdx = i;
    //     }
    // }

    return vec4<f32>(finalColor, 1.0);
}