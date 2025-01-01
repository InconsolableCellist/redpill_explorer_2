import * as THREE from 'three';
export class SpicyVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.nodes = new Map(); // Map of hash to THREE.Mesh
        this.nodeData = new Map(); // Map of hash to data object
        this.selectedTags = new Set();
        this.tagToColorIndex = new Map(); // Maps selected tags to their color index
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.COLUMN_HEIGHT = 1000;
        this.SLICE_RADIUS = 200;
        this.centerOffset = 0;
        this.sliceView = null;

        // Predefined color palette for tag selection
        this.TAG_COLORS = [
            new THREE.Color('#ff4444'), // Red
            new THREE.Color('#44ff44'), // Green
            new THREE.Color('#4444ff'), // Blue
            new THREE.Color('#ffff44'), // Yellow
            new THREE.Color('#ff44ff'), // Magenta
            new THREE.Color('#44ffff'), // Cyan
            new THREE.Color('#ff8844'), // Orange
            new THREE.Color('#88ff44'), // Lime
            new THREE.Color('#4488ff'), // Light Blue
            new THREE.Color('#ff4488')  // Pink
        ];

        // Add properties for hover highlighting
        this.highlightMaterial = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            emissive: 0x444444,
            transparent: true,
            opacity: 1
        });
        this.hoveredNode = null;
        this.originalMaterial = null;
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
        this.centerOffset = -this.SLICE_RADIUS - 50;
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
        // Create cylindrical slice geometry for reuse
        const sliceGeometry = new THREE.CylinderGeometry(this.SLICE_RADIUS, this.SLICE_RADIUS, 50, 64, 1, true);

        // Group nodes by spiciness level
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

            // Create slice cylinder
            const sliceMaterial = new THREE.MeshPhongMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.1,
                side: THREE.DoubleSide
            });
            const sliceMesh = new THREE.Mesh(sliceGeometry, sliceMaterial);
            sliceMesh.position.set(0, y, 0);
            sliceMesh.userData = {
                type: 'slice',
                level: spicyLevel,
                items: items
            };
            this.scene.add(sliceMesh);

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
            label.position.set(this.centerOffset, y, 0);
            label.scale.set(100, 25, 1);
            this.scene.add(label);

            // Create nodes within the slice
            items.forEach((item) => {
                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * this.SLICE_RADIUS;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius;

                const geometry = new THREE.SphereGeometry(5, 32, 32);
                const material = new THREE.MeshPhongMaterial({
                    color: 0x888888,  // Default grey color
                    transparent: true,
                    opacity: 1
                });

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
            // Reset all nodes to default grey
            this.nodes.forEach((node) => {
                node.material.color.setHex(0x888888);
                node.material.opacity = 1;
                node.material.transparent = false;
            });
            this.tagToColorIndex.clear();
            return;
        }

        // Update color mappings for selected tags
        Array.from(this.selectedTags).forEach((tag, index) => {
            if (!this.tagToColorIndex.has(tag)) {
                this.tagToColorIndex.set(tag, index % this.TAG_COLORS.length);
            }
        });

        // Update node appearances
        this.nodes.forEach((node, hash) => {
            const itemData = this.nodeData.get(hash);
            const nodeTags = new Set(Object.keys(itemData.tags));

            // Check if node has ALL selected tags (AND operation)
            const hasAllTags = Array.from(this.selectedTags)
                .every(tag => nodeTags.has(tag));

            if (hasAllTags) {
                // If node has all tags, use the color of the first tag
                const firstTag = Array.from(this.selectedTags)[0];
                const colorIndex = this.tagToColorIndex.get(firstTag);
                node.material.color.copy(this.TAG_COLORS[colorIndex]);
                node.material.opacity = 1;
                node.material.transparent = false;
            } else {
                // If node doesn't have all tags, make it semi-transparent grey
                node.material.color.setHex(0x888888);
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

        // Filter for objects that have userData.type
        for (const intersect of intersects) {
            if (intersect.object.userData &&
                (intersect.object.userData.type === 'node' ||
                    intersect.object.userData.type === 'slice')) {
                return intersect.object;
            }
        }
        return null;
    }

    highlightNode(node) {
        if (this.hoveredNode === node) return;

        // Reset previous highlight
        this.unhighlightNode();

        if (node) {
            // Store current material and apply highlight
            this.hoveredNode = node;
            this.originalMaterial = node.material;
            node.material = this.highlightMaterial.clone();
            node.material.color = this.originalMaterial.color.clone();
            node.scale.set(1.5, 1.5, 1.5); // Make node bigger
        }
    }

    unhighlightNode() {
        if (this.hoveredNode) {
            this.hoveredNode.material = this.originalMaterial;
            this.hoveredNode.scale.set(1, 1, 1); // Reset size
            this.hoveredNode = null;
            this.originalMaterial = null;
        }
    }
}