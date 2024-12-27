// Core visualization functionality
export class SpicyVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.nodes = new Map(); // Map of hash to THREE.Mesh
        this.nodeData = new Map(); // Map of hash to data object
        this.selectedTags = new Set();
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.COLUMN_HEIGHT = 1000;
        this.SLICE_RADIUS = 200;
    }

    async initialize() {
        // Load data
        const response = await fetch('/data');
        const data = await response.json();

        // Setup Three.js scene
        this.setupScene();
        this.createNodes(data);
        this.setupCamera();
        this.setupLights();
        this.setupRenderer();
        this.animate();

        // Store data for later use
        Object.entries(data).forEach(([hash, itemData]) => {
            this.nodeData.set(hash, itemData);
        });

        return this;
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
    }

    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        this.camera.position.set(400, 500, 400);
        this.camera.lookAt(0, this.COLUMN_HEIGHT / 2, 0);
    }

    setupLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 0);
        this.scene.add(directionalLight);
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(this.renderer.domElement);
    }

    createNodes(data) {
        // Group nodes by spiciness level (rounded to 1 decimal)
        const spicyLevels = new Map();

        Object.entries(data).forEach(([hash, itemData]) => {
            const spicyLevel = Math.round(itemData.spicy * 10) / 10;
            if (!spicyLevels.has(spicyLevel)) {
                spicyLevels.set(spicyLevel, []);
            }
            spicyLevels.get(spicyLevel).push({ hash, ...itemData });
        });

        // Create nodes for each spiciness level
        spicyLevels.forEach((items, spicyLevel) => {
            const y = spicyLevel * this.COLUMN_HEIGHT;

            items.forEach((item) => {
                // Random position within the slice
                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * this.SLICE_RADIUS;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius;

                // Create node geometry
                const geometry = new THREE.SphereGeometry(5, 32, 32);
                const material = new THREE.MeshPhongMaterial({ color: 0xffffff });
                const node = new THREE.Mesh(geometry, material);

                node.position.set(x, y, z);
                node.userData.hash = item.hash;

                this.scene.add(node);
                this.nodes.set(item.hash, node);
            });
        });
    }

    updateSelection() {
        if (this.selectedTags.size === 0) {
            // Reset all nodes to visible and white
            this.nodes.forEach((node) => {
                node.material.color.setHex(0xffffff);
                node.material.opacity = 1;
                node.material.transparent = false;
            });
            return;
        }

        // Update visibility based on selected tags
        this.nodes.forEach((node, hash) => {
            const itemData = this.nodeData.get(hash);
            const hasAllTags = Array.from(this.selectedTags)
                .every(tag => itemData.tags && itemData.tags[tag] !== undefined);

            if (hasAllTags) {
                node.material.color.setHex(0xff0000);
                node.material.opacity = 1;
                node.material.transparent = false;
            } else {
                node.material.color.setHex(0x333333);
                node.material.opacity = 0.2;
                node.material.transparent = true;
            }
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    getIntersectedNode(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children);

        if (intersects.length > 0) {
            return intersects[0].object;
        }
        return null;
    }
}