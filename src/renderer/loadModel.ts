import { WebIO } from "@gltf-transform/core";

export async function loadModel(modelUri: string) {
    const io = new WebIO({ credentials: "include" });
    const doc = await io.read(modelUri);

    const positions = doc
        .getRoot()
        .listMeshes()[0]
        .listPrimitives()[0]
        .getAttribute("POSITION")
        ?.getArray();
    const indices = doc
        .getRoot()
        .listMeshes()[0]
        .listPrimitives()[0]
        .getIndices()
        ?.getArray();

    const uvs = doc
        .getRoot()
        .listMeshes()[0]
        .listPrimitives()[0]
        .getAttribute("TEXCOORD_0")
        ?.getArray();

    const textureUri = doc
        .getRoot()
        .listMaterials()[0]
        .getBaseColorTexture()
        ?.getURI();

    if (textureUri == null) {
        throw new Error("Texture is null");
    }

    const basePath = modelUri.slice(0, modelUri.lastIndexOf("/"));

    let textureBitmap = null;
    try {
        const response = await fetch(basePath + "/" + textureUri);
        const textureBlob = await response.blob();
        textureBitmap = await createImageBitmap(textureBlob, {});
    } catch {
        throw new Error("cannot load texture: " + basePath + "/" + textureUri);
    }

    const finalData = [];

    if (indices == null) {
        throw Error("indices are null :(");
    } else if (positions == null || uvs == null) {
        throw Error("positions or uvs are null :(");
    }

    for (let i = 0; i < indices.length; i++) {
        const posIndex1 = indices[i] * 3 + 0;
        const posIndex2 = indices[i] * 3 + 1;
        const posIndex3 = indices[i] * 3 + 2;
        const uvIndex1 = indices[i] * 2 + 0;
        const uvIndex2 = indices[i] * 2 + 1;

        finalData.push(positions[posIndex1]);
        finalData.push(positions[posIndex2]);
        finalData.push(positions[posIndex3]);
        finalData.push(uvs[uvIndex1]);
        finalData.push(uvs[uvIndex2]);
    }
    return { vertexData: finalData, baseColor: textureBitmap };
}
