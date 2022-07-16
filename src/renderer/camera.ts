import { mat3, mat4, quat, vec3 } from "gl-matrix";

export class Camera {
    private rotationDeg: [number, number, number];
    private rotationQuat: quat;
    private projectionMatrix: mat4;

    readonly fovY: number;
    readonly near: number;
    readonly far: number;
    position: vec3;

    constructor(
        fovY: number,
        aspectRatio: number,
        near: number = 0.1,
        far: number = 1000
    ) {
        this.position = [0, 0, 0];
        this.rotationDeg = [0, 0, 0];
        this.rotationQuat = quat.create();
        this.projectionMatrix = mat4.create();
        this.fovY = fovY;
        this.near = near;
        this.far = far;
        mat4.perspectiveZO(this.projectionMatrix, fovY, aspectRatio, near, far);
    }

    set aspectRatio(ratio: number) {
        mat4.perspectiveZO(
            this.projectionMatrix,
            this.fovY,
            ratio,
            this.near,
            this.far
        );
    }

    activate() {}

    deactivate() {}

    get rotation() {
        return [...this.rotationDeg];
    }

    set rotation(rotationDeg: [number, number, number]) {
        this.rotationDeg = [...rotationDeg];
        quat.fromEuler(this.rotationQuat, ...rotationDeg);
    }

    copyTransform(other: Camera) {
        this.rotation = other.rotation;
        this.position = vec3.clone(other.position);
    }

    getUpRightMatrix(): mat3 {
        const result = mat3.create();
        mat3.fromQuat(result, this.rotationQuat);
        return result;
    }

    getViewMatrix(): mat4 {
        const result = mat4.create();
        mat4.fromQuat(result, this.rotationQuat);
        mat4.transpose(result, result);
        const negPosition = vec3.create();
        vec3.negate(negPosition, this.position);
        mat4.translate(result, result, negPosition);
        return result;
    }

    getViewProjectionMatrix(): mat4 {
        const result = mat4.create();
        mat4.mul(result, this.projectionMatrix, this.getViewMatrix());
        return result;
    }
}

export class FreeControlledCamera extends Camera {
    moveSpeed: number = 0.1;
    rotationSpeed: number = 0.1;
    rotateOnlyIfFocussed: boolean = true;

    private readonly state = { forward: 0, right: 0 };
    private hasFocus: boolean = false;
    private readonly canvas: HTMLCanvasElement;
    private lockMouseOnClickHandle: () => void;

    constructor(
        canvas: HTMLCanvasElement,
        fovY: number,
        aspectRatio: number,
        near: number = 0.1,
        far: number = 1000
    ) {
        super(fovY, aspectRatio, near, far);

        this.canvas = canvas;
        this.lockMouseOnClickHandle = this.lockMouseOnClick.bind(this);
        canvas.setAttribute("tabindex", "0");

        const handler = (event: KeyboardEvent) => {
            if (event.repeat) {
                return;
            }

            const dir = event.type === "keydown" ? 1 : -1;
            if (event.key === "w" && dir * this.state.forward >= 0) {
                this.state.forward -= dir * this.moveSpeed;
            }
            if (event.key === "s" && dir * this.state.forward <= 0) {
                this.state.forward += dir * this.moveSpeed;
            }
            if (event.key === "d" && dir * this.state.right <= 0) {
                this.state.right += dir * this.moveSpeed;
            }
            if (event.key === "a" && dir * this.state.right >= 0) {
                this.state.right -= dir * this.moveSpeed;
            }
        };

        canvas.addEventListener("keyup", handler, true);
        canvas.addEventListener("keydown", handler, true);

        canvas.addEventListener(
            "mousemove",
            (event) => {
                if (this.hasFocus || !this.rotateOnlyIfFocussed) {
                    const currentRotation = this.rotation;
                    currentRotation[1] -= event.movementX * this.rotationSpeed;
                    currentRotation[0] -= event.movementY * this.rotationSpeed;
                    this.rotation = currentRotation;
                }
            },
            false
        );

        document.addEventListener("pointerlockchange", () => {
            this.hasFocus = document.pointerLockElement === canvas;
        });
    }

    private lockMouseOnClick() {
        this.canvas.requestPointerLock();
    }

    override activate() {
        this.canvas.addEventListener("click", this.lockMouseOnClickHandle);
    }

    override deactivate() {
        this.canvas.removeEventListener("click", this.lockMouseOnClickHandle);
    }

    updateAndGetViewMatrix(): mat4 {
        const mat = this.getUpRightMatrix();
        const forward: vec3 = [mat[6], mat[7], mat[8]];
        const right: vec3 = [mat[0], mat[1], mat[2]];

        vec3.scale(forward, forward, this.state.forward);
        vec3.scale(right, right, this.state.right);
        vec3.add(right, right, forward);
        vec3.add(this.position, this.position, right);

        return this.getViewMatrix();
    }

    updateAndGetViewProjectionMatrix(): mat4 {
        this.updateAndGetViewMatrix();
        return this.getViewProjectionMatrix();
    }
}

export class TurnTableCamera extends Camera {
    rotationPivot = vec3.fromValues(0, 10, 0);
    rotationSpeed = 0.01;
    rotatationRadius = 30;
    lookAt = vec3.fromValues(0, 0, 0);

    private angleRad = 0;

    constructor(
        fovY: number,
        aspectRatio: number,
        near: number = 0.1,
        far: number = 1000
    ) {
        super(fovY, aspectRatio, near, far);
    }

    updateAndGetViewMatrix(): mat4 {
        this.angleRad += this.rotationSpeed;
        this.position[0] = this.rotatationRadius * Math.sin(this.angleRad);
        this.position[2] = this.rotatationRadius * Math.cos(this.angleRad);
        this.position[1] = this.rotationPivot[1];

        const currentRoation = this.rotation;
        currentRoation[1] = (180 * this.angleRad) / Math.PI;

        const direction = vec3.create();
        vec3.sub(direction, this.lookAt, this.position);
        vec3.normalize(direction, direction);

        const up = vec3.fromValues(0, 1, 0);
        const right = vec3.create();
        vec3.cross(right, direction, up);

        const slope = vec3.create();
        vec3.cross(slope, right, direction);
        vec3.normalize(slope, slope);

        const angle = -Math.acos(vec3.dot(slope, up));
        currentRoation[0] = (180 * angle) / Math.PI;

        this.rotation = currentRoation;

        return this.getViewMatrix();
    }

    updateAndGetViewProjectionMatrix(): mat4 {
        this.updateAndGetViewMatrix();
        return this.getViewProjectionMatrix();
    }
}
