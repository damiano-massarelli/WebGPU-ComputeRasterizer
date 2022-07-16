import { run } from "./renderer/renderer";

if (navigator.gpu) {
    run();
} else {
    document.getElementById("webgpu-available")!.innerText =
        "Looks like webgpu is not available :(";
}
