import shader from "./shaders.wgsl";
import * as dat from "dat.gui";
import { mat4, vec3, vec4 } from "gl-matrix";
import { FreeControlledCamera } from "./camera";
import { loadModel } from "./loadModel";

const USE_DEVICE_PIXEL_RATIO = true;

interface IRenderingContext {
    device: GPUDevice;
    canvas: HTMLCanvasElement;
    context: GPUCanvasContext;
    presentationSize: readonly [number, number];
    presentationFormat: GPUTextureFormat;
}

async function init(
    canvasId: string,
    useDevicePixelRatio: boolean
): Promise<IRenderingContext> {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;

    const context = canvas.getContext("webgpu");

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();

    const devicePixelRatio = useDevicePixelRatio
        ? window.devicePixelRatio ?? 1
        : 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio,
    ] as const;

    canvas.width = presentationSize[0];
    canvas.height = presentationSize[1];

    const presentationFormat: GPUTextureFormat =
        navigator.gpu.getPreferredCanvasFormat();
    console.log(presentationFormat);
    context?.configure({
        device,
        format: presentationFormat,
        alphaMode: "opaque",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    return {
        device,
        canvas,
        context: context!,
        presentationSize,
        presentationFormat,
    };
}

function createFullscreenPass(
    context: IRenderingContext,
    colorBuffer: GPUBuffer,
    colorBuffer2: GPUBuffer,
    uniforms: GPUBuffer
) {
    const fullscreenBindGroupLayout = context.device.createBindGroupLayout({
        label: "full screen bind group layout",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT, // color buffer 1
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT, // color buffer 2
                buffer: {
                    type: "read-only-storage",
                },
            },
        ],
    });

    const fullScreenPipeline = context.device.createRenderPipeline({
        label: "full screen pipeline",
        layout: context.device.createPipelineLayout({
            bindGroupLayouts: [fullscreenBindGroupLayout],
        }),
        vertex: {
            module: context.device.createShaderModule({
                code: shader,
            }),
            entryPoint: "fullScreenVert",
        },
        fragment: {
            module: context.device.createShaderModule({ code: shader }),
            entryPoint: "fullScreenFrag",
            targets: [
                {
                    format: context.presentationFormat,
                },
            ],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    const fullscreenBindGroup = context.device.createBindGroup({
        layout: fullscreenBindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniforms,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: colorBuffer,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: colorBuffer2,
                },
            },
        ],
    });

    const addFullscreenPass = (
        context: IRenderingContext,
        commandEncoder: GPUCommandEncoder
    ) => {
        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.context.getCurrentTexture().createView(),
                    clearValue: [1, 1, 1, 1],
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });

        passEncoder.setPipeline(fullScreenPipeline);
        passEncoder.setBindGroup(0, fullscreenBindGroup);
        passEncoder.draw(6, 1, 0, 0);
        passEncoder.end();
    };

    return { addFullscreenPass };
}

function createComputePass(
    context: IRenderingContext,
    vertexBuffer: number[],
    stride: number,
    baseColor: ImageBitmap
) {
    if (stride <= 0 || vertexBuffer.length % stride != 0) {
        throw new Error(
            "Wrong stride " +
                stride +
                "for vertex buffer of length " +
                vertexBuffer.length
        );
    }

    // Base color texture
    const textureDescriptor: GPUTextureDescriptor = {
        size: { width: baseColor.width, height: baseColor.height },
        format: "rgba8unorm",
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
    };
    const baseColorTexture = context.device.createTexture(textureDescriptor);
    context.device.queue.copyExternalImageToTexture(
        { source: baseColor },
        { texture: baseColorTexture },
        textureDescriptor.size
    );

    const WIDTH = context.presentationSize[0];
    const HEIGHT = context.presentationSize[1];

    const gpuVertexBuffer = context.device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * vertexBuffer.length,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        label: "gpu vertex buffer",
        mappedAtCreation: true,
    });
    new Float32Array(gpuVertexBuffer.getMappedRange()).set(vertexBuffer);
    gpuVertexBuffer.unmap();

    const outputColorBufferSize =
        Uint32Array.BYTES_PER_ELEMENT * (WIDTH * HEIGHT);

    const colorBuffer1 = context.device.createBuffer({
        label: "output color buffer 1",
        size: outputColorBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const colorBuffer2 = context.device.createBuffer({
        label: "output depth buffer",
        size: outputColorBufferSize,
        usage: GPUBufferUsage.STORAGE,
    });

    const UBOBufferSize = (16 + 16 + 4) * 4; // view projection + model + width + height + num_tris + stride
    const UBOBuffer = context.device.createBuffer({
        label: "uniform buffer",
        size: UBOBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(UBOBuffer.getMappedRange()).set([
        ...mat4.create(),
        ...mat4.create(),
        WIDTH,
        HEIGHT,
        vertexBuffer.length / 3 / stride, // three vertices per triangle with stride data
        stride,
    ]);
    UBOBuffer.unmap();

    const bindGroupLayout = context.device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE, // color buffer 1
                buffer: {
                    type: "storage",
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE, // color buffer 2
                buffer: {
                    type: "storage",
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE, // uniform
                buffer: {
                    type: "uniform",
                },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE, // vertex buffer
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                texture: {
                    viewDimension: "2d",
                },
            },
        ],
    });

    const bindGroup = context.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: colorBuffer1,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: colorBuffer2,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: UBOBuffer,
                },
            },
            {
                binding: 3,
                resource: {
                    buffer: gpuVertexBuffer,
                },
            },
            {
                binding: 4,
                resource: baseColorTexture.createView(),
            },
        ],
    });

    const clearPipeline = context.device.createComputePipeline({
        label: "clear rasterizer",
        compute: {
            module: context.device.createShaderModule({
                code: shader,
            }),
            entryPoint: "clear",
        },
        layout: context.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        }),
    });

    const rasterizerPipeline = context.device.createComputePipeline({
        label: "compute rasterizer",
        compute: {
            module: context.device.createShaderModule({
                code: shader,
            }),
            entryPoint: "rasterizer",
        },
        layout: context.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        }),
    });

    const addComputePass = (
        commandEncoder: GPUCommandEncoder,
        viewProjection: mat4,
        model: mat4
    ) => {
        context.device.queue.writeBuffer(
            UBOBuffer,
            0,
            new Float32Array([...viewProjection, ...model])
        );
        const clearPassEncorder = commandEncoder.beginComputePass();
        clearPassEncorder.setPipeline(clearPipeline);
        clearPassEncorder.setBindGroup(0, bindGroup);
        clearPassEncorder.dispatchWorkgroups(
            Math.ceil(context.presentationSize[0] / 16),
            Math.ceil(context.presentationSize[1] / 16)
        );
        clearPassEncorder.end();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(rasterizerPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(vertexBuffer.length / stride / 256)
        );
        passEncoder.end();

        // bytesPerRow must be a multiple of 256, if presentation format is not rgb colors will be messed up
        // commandEncoder.copyBufferToTexture(
        //     { buffer: outputColorBuffer, bytesPerRow: 4 * WIDTH },
        //     { texture: context.context.getCurrentTexture() },
        //     { width: WIDTH, height: HEIGHT }
        // );
    };

    return {
        addComputePass,
        colorBuffer1,
        colorBuffer2,
        UBOBuffer,
    };
}

export async function run() {
    const gui = new dat.GUI();

    if (!("gpu" in navigator)) {
        return;
    }

    const { vertexData, baseColor } = await loadModel("models/dragon.gltf");

    const context = await init("canvas-wegbpu", USE_DEVICE_PIXEL_RATIO);

    const camera = new FreeControlledCamera(
        context.canvas,
        (2 * Math.PI) / 5,
        context.presentationSize[0] / context.presentationSize[1],
        0.1,
        150
    );
    camera.activate();

    const { addComputePass, colorBuffer1, colorBuffer2, UBOBuffer } =
        createComputePass(context, vertexData, 5, baseColor);

    const { addFullscreenPass } = createFullscreenPass(
        context,
        colorBuffer1,
        colorBuffer2,
        UBOBuffer
    );

    let t = 0;
    function frame() {
        const commandEncoder = context.device.createCommandEncoder();

        let modelMatrix = mat4.create();
        mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0, 10, -50));
        mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(0.3, -0.3, 0.3));
        mat4.rotate(
            modelMatrix,
            modelMatrix,
            0.01 * t,
            vec3.fromValues(0, 1, 0)
        );
        // mat4.rotate(
        //     modelMatrix,
        //     modelMatrix,
        //     0.1 * t,
        //     vec3.fromValues(0, 0, 1)
        // );
        t++;

        addComputePass(
            commandEncoder,
            camera.updateAndGetViewProjectionMatrix(),
            modelMatrix
        );
        addFullscreenPass(context, commandEncoder);

        context.device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}
