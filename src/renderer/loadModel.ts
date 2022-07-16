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
    const finalPositions = [];

    if (indices == null) {
        throw Error("indices are null :(");
    } else if (positions == null) {
        throw Error("positions are null :(");
    }

    for (let i = 0; i < indices.length; i++) {
        const index1 = indices[i] * 3 + 0;
        const index2 = indices[i] * 3 + 1;
        const index3 = indices[i] * 3 + 2;

        finalPositions.push(positions[index1]);
        finalPositions.push(positions[index2]);
        finalPositions.push(positions[index3]);
    }
    return finalPositions;
}
