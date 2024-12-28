import * as THREE from 'three';
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
        this.centerOffset = 0; // Will be calculated based on viewport
        this.sliceView = null;
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
        this.centerOffset = -this.SLICE_RADIUS - 50; // Adjust based on label width
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
        // Generate a color map for tags
        this.tagColors = new Map();
        const allTags = new Set();
        Object.values(data).forEach(item => {
            Object.keys(item.tags).forEach(tag => allTags.add(tag));
        });

        Array.from(allTags).forEach(tag => {
            this.tagColors.set(tag, new THREE.Color(Math.random(), Math.random(), Math.random()));
        });

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

            // Create label for this slice
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'white';
            ctx.font = 'bold 32px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`Spicy: ${spicyLevel.toFixed(1)}`, canvas.width/2, canvas.height/2);

            const texture = new THREE.CanvasTexture(canvas);
            const labelMaterial = new THREE.SpriteMaterial({ map: texture });
            const label = new THREE.Sprite(labelMaterial);
            label.position.set(-this.SLICE_RADIUS - 50, y, 0);
            label.scale.set(100, 25, 1);
            this.scene.add(label);

            items.forEach((item) => {
                // Random position within the slice
                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * this.SLICE_RADIUS;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius;

                // Create node geometry
                const geometry = new THREE.SphereGeometry(5, 32, 32);

                // Get color from first tag (or white if no tags)
                const firstTag = Object.keys(item.tags)[0];
                const color = firstTag ? this.tagColors.get(firstTag) : new THREE.Color(0xffffff);
                const material = new THREE.MeshPhongMaterial({ color });

                const node = new THREE.Mesh(geometry, material);
                node.position.set(x, y, z);
                node.userData = {
                    type: 'node',
                    hash: item.hash,
                    tags: Object.keys(item.tags),
                    spicyLevel,
                    radius,
                    angle
                };

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