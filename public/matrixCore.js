// Core matrix visualization functionality
export class MatrixVisualizer {
    constructor() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.CELL_SIZE = 40;
        this.LABEL_MARGIN = 120;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.raycaster = null;
        this.mouse = null;
        this.matrixMesh = null;
        this.tagList = [];
        this.tagPairs = null;
        this.selectedRow = -1;
        this.selectedCol = -1;
        this.isDragging = false;
        this.previousMousePosition = { x: 0, y: 0 };
        this.uniforms = null;
    }

    async initialize() {
        // Load tag data
        const response = await fetch('/tag_pairs_with_weights');
        this.tagPairs = await response.json();
        this.tagList = Array.from(new Set([
            ...Object.keys(this.tagPairs),
            ...Object.values(this.tagPairs).flatMap(obj => Object.keys(obj))
        ])).sort();

        this.setupScene();
        this.setupMatrix();
        this.setupLabels();
        this.setupEventListeners();
        this.animate();
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        const frustumSize = 1000;
        const aspect = this.width / this.height;
        this.camera = new THREE.OrthographicCamera(
            frustumSize * aspect / -2,
            frustumSize * aspect / 2,
            frustumSize / 2,
            frustumSize / -2,
            1,
            2000
        );
        this.camera.position.z = 1000;

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.width, this.height);
        document.body.appendChild(this.renderer.domElement);

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
    }

    setupMatrix() {
        const matrixSize = this.tagList.length;
        const matrix = new Float32Array(matrixSize * matrixSize);
        let maxWeight = 0;

        for (let i = 0; i < matrixSize; i++) {
            for (let j = 0; j < matrixSize; j++) {
                if (i === j) {
                    matrix[i * matrixSize + j] = 1.0;
                } else {
                    const fromTag = this.tagList[i];
                    const toTag = this.tagList[j];
                    const weight = this.tagPairs[fromTag]?.[toTag] ?? 0.0;
                    matrix[i * matrixSize + j] = weight;
                    maxWeight = Math.max(maxWeight, weight);
                }
            }
        }

        const geometry = new THREE.PlaneGeometry(
            matrixSize * this.CELL_SIZE,
            matrixSize * this.CELL_SIZE
        );

        const texture = new THREE.DataTexture(
            matrix,
            matrixSize,
            matrixSize,
            THREE.RedFormat,
            THREE.FloatType
        );
        texture.needsUpdate = true;

        this.uniforms = {
            matrix: { value: texture },
            maxWeight: { value: maxWeight },
            selectedRow: { value: -1 },
            selectedCol: { value: -1 }
        };

        const material = new THREE.ShaderMaterial({
            uniforms: this.uniforms,
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D matrix;
                uniform float maxWeight;
                uniform int selectedRow;
                uniform int selectedCol;
                varying vec2 vUv;

                void main() {
                    float weight = texture2D(matrix, vUv).r;
                    vec2 coord = vUv * ${matrixSize}.0;
                    int row = int(coord.y);
                    int col = int(coord.x);
                    
                    vec3 color;
                    if (selectedRow >= 0 || selectedCol >= 0) {
                        if (row == selectedRow || col == selectedCol) {
                            color = mix(
                                vec3(1.0, 1.0, 1.0),
                                vec3(1.0, 0.0, 0.0),
                                weight
                            );
                        } else {
                            color = mix(
                                vec3(1.0, 1.0, 1.0),
                                vec3(0.8, 0.8, 0.8),
                                weight
                            );
                        }
                    } else {
                        color = mix(
                            vec3(1.0, 1.0, 1.0),
                            vec3(1.0, 0.0, 0.0),
                            weight
                        );
                    }
                    gl_FragColor = vec4(color, 1.0);
                }
            `
        });

        this.matrixMesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.matrixMesh);
    }

    createTextSprite(text, rotate = false) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        if (rotate) {
            canvas.width = 128;
            canvas.height = 256;
        } else {
            canvas.width = 256;
            canvas.height = 64;
        }
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = 'black';
        ctx.font = rotate ? '24px Arial' : '24px Arial';  // Increased font size
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        if (rotate) {
            ctx.translate(canvas.width/2, canvas.height/2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(text, 0, 0);
        } else {
            ctx.fillText(text, canvas.width/2, canvas.height/2);
        }

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        
        if (rotate) {
            sprite.scale.set(this.LABEL_MARGIN * 0.5, this.LABEL_MARGIN, 1);
        } else {
            sprite.scale.set(this.LABEL_MARGIN * 1.2, this.LABEL_MARGIN * 0.3, 1);
        }
        
        return sprite;
    }

    setupLabels() {
        const matrixSize = this.tagList.length;

        // Row labels
        this.tagList.forEach((tag, i) => {
            const label = this.createTextSprite(tag);
            const y = (matrixSize/2 - (matrixSize - 1 - i) - 0.5) * this.CELL_SIZE;
            label.position.set(-matrixSize * this.CELL_SIZE / 2 - this.LABEL_MARGIN/2, y, 0);
            this.scene.add(label);
        });

        // Column labels
        this.tagList.forEach((tag, i) => {
            const label = this.createTextSprite(tag, true);
            const x = (-matrixSize / 2 + i + 0.5) * this.CELL_SIZE;
            label.position.set(x, matrixSize * this.CELL_SIZE / 2 + this.LABEL_MARGIN, 0);
            this.scene.add(label);
        });
    }

    getIntersectedCell(event) {
        this.mouse.x = (event.clientX / this.width) * 2 - 1;
        this.mouse.y = -(event.clientY / this.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.matrixMesh);

        if (intersects.length > 0) {
            const uv = intersects[0].uv;
            const matrixSize = this.tagList.length;
            const x = Math.floor(uv.x * matrixSize);
            const y = Math.floor(uv.y * matrixSize);
            return { row: y, col: x };
        }
        return null;
    }

    setSelection(row, col) {
        this.selectedRow = row;
        this.selectedCol = col;
        this.uniforms.selectedRow.value = row;
        this.uniforms.selectedCol.value = col;
    }

    clearSelection() {
        this.setSelection(-1, -1);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        
        const aspect = this.width / this.height;
        const matrixSize = this.tagList.length * this.CELL_SIZE;
        const padding = matrixSize * 0.2;
        
        if (aspect > 1) {
            this.camera.left = -(matrixSize + padding + this.LABEL_MARGIN) * aspect;
            this.camera.right = (matrixSize + padding + this.LABEL_MARGIN) * aspect;
            this.camera.top = matrixSize + padding + this.LABEL_MARGIN;
            this.camera.bottom = -(matrixSize + padding + this.LABEL_MARGIN);
        } else {
            this.camera.left = -(matrixSize + padding + this.LABEL_MARGIN);
            this.camera.right = matrixSize + padding + this.LABEL_MARGIN;
            this.camera.top = (matrixSize + padding + this.LABEL_MARGIN) / aspect;
            this.camera.bottom = -(matrixSize + padding + this.LABEL_MARGIN) / aspect;
        }
        
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }
}
