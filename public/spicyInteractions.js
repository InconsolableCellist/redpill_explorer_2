import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class SpicyInteractions {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.controls = null;
        this.tooltip = document.getElementById('tooltip');
        this.isTooltipVisible = false;
        this.setupOrbitControls();
        this.setupEventListeners();
        this.setupTagPanel();
        this.selectedSlice = null;
        this.hoveredSlice = null;
        this.setupSliceViewer();
    }

    setupOrbitControls() {
        // Initialize OrbitControls
        this.controls = new OrbitControls(
            this.visualizer.camera,
            this.visualizer.renderer.domElement
        );

        // Configure basic controls
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 1500;

        // Disable built-in controls - we'll handle them manually
        this.controls.enableRotate = false;
        this.controls.enablePan = false;

        let isLeftMouseDown = false;
        let isRightMouseDown = false;
        let previousX = 0;
        let previousY = 0;
        let cameraAngle = 0; // Track camera's vertical angle

        this.visualizer.renderer.domElement.addEventListener('mousedown', (event) => {
            if (event.button === 0) { // Left button
                isLeftMouseDown = true;
                previousX = event.clientX;
                previousY = event.clientY;
            } else if (event.button === 2) { // Right button
                isRightMouseDown = true;
                previousY = event.clientY;
            }
        });

        this.visualizer.renderer.domElement.addEventListener('mousemove', (event) => {
            if (isLeftMouseDown) {
                // Horizontal movement - orbit around vertical axis
                const deltaX = event.clientX - previousX;
                const rotationSpeed = 0.01;
                const radius = Math.sqrt(
                    Math.pow(this.visualizer.camera.position.x, 2) +
                    Math.pow(this.visualizer.camera.position.z, 2)
                );

                const currentAngle = Math.atan2(
                    this.visualizer.camera.position.z,
                    this.visualizer.camera.position.x
                );
                const newAngle = currentAngle - deltaX * rotationSpeed;

                this.visualizer.camera.position.x = radius * Math.cos(newAngle);
                this.visualizer.camera.position.z = radius * Math.sin(newAngle);

                // Vertical movement - translate up/down
                const deltaY = event.clientY - previousY;
                this.visualizer.camera.position.y += deltaY * 2;
                this.controls.target.y = this.visualizer.camera.position.y;

                previousX = event.clientX;
                previousY = event.clientY;
            }

            if (isRightMouseDown) {
                // Vertical camera rotation around X axis
                const deltaY = event.clientY - previousY;
                const rotationSpeed = 0.01;

                cameraAngle = Math.max(Math.min(cameraAngle + deltaY * rotationSpeed, Math.PI/3), -Math.PI/3);

                // Get current radius in XZ plane
                const radius = Math.sqrt(
                    Math.pow(this.visualizer.camera.position.x, 2) +
                    Math.pow(this.visualizer.camera.position.z, 2)
                );

                // Adjust camera height based on angle while maintaining distance
                const currentY = this.visualizer.camera.position.y;
                const baseY = this.controls.target.y;
                const heightOffset = radius * Math.sin(cameraAngle);
                this.visualizer.camera.position.y = baseY + heightOffset;

                previousY = event.clientY;
            }

            // Always look at target
            this.visualizer.camera.lookAt(this.controls.target);
        });

        document.addEventListener('mouseup', (event) => {
            if (event.button === 0) {
                isLeftMouseDown = false;
            } else if (event.button === 2) {
                isRightMouseDown = false;
            }
        });

        // Prevent context menu
        this.visualizer.renderer.domElement.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });

        // Update controls in animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            this.controls.update();
        };
        animate();
    }

    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => {
            this.visualizer.onWindowResize();
        }, false);

        // Mouse events for node and slice interaction
        this.visualizer.renderer.domElement.addEventListener('mousemove', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);

            // Reset previous hover state
            if (this.hoveredSlice) {
                this.hoveredSlice.material.opacity = 0.1;
                this.hoveredSlice = null;
            }

            if (intersectedObject) {
                if (intersectedObject.userData.type === 'slice') {
                    // Highlight hovered slice
                    intersectedObject.material.opacity = 0.3;
                    this.hoveredSlice = intersectedObject;
                    this.updateTooltip(event, null); // Hide node tooltip when hovering slice
                } else if (intersectedObject.userData.type === 'node') {
                    this.updateTooltip(event, intersectedObject);
                }
            } else {
                this.hideTooltip();
            }
        });

        // Hide tooltip when mouse leaves canvas
        this.visualizer.renderer.domElement.addEventListener('mouseleave', () => {
            this.hideTooltip();
            if (this.hoveredSlice) {
                this.hoveredSlice.material.opacity = 0.1;
                this.hoveredSlice = null;
            }
        });

        // Click handling
        this.visualizer.renderer.domElement.addEventListener('click', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);
            if (!intersectedObject) return;

            if (intersectedObject.userData.type === 'slice') {
                // Deselect previous slice if different
                if (this.selectedSlice && this.selectedSlice !== intersectedObject) {
                    this.selectedSlice.material.opacity = 0.1;
                }

                // Select new slice
                this.selectedSlice = intersectedObject;
                intersectedObject.material.opacity = 0.4;
                this.showSliceView(intersectedObject);
            } else if (intersectedObject.userData.type === 'node') {
                // Single click selects the first tag
                if (intersectedObject.userData.tags.length > 0) {
                    const tag = intersectedObject.userData.tags[0];
                    this.visualizer.selectedTags.add(tag);
                    this.updateSelectedTagsPanel();
                    this.visualizer.updateSelection();
                }
            }
        });

        // Double click handling
        this.visualizer.renderer.domElement.addEventListener('dblclick', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);
            if (intersectedObject && intersectedObject.userData.type === 'node') {
                window.open(`/node-info/${intersectedObject.userData.hash}`, '_blank');
            }
        });

        // Keyboard events
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                if (this.selectedSlice) {
                    this.selectedSlice.material.opacity = 0.1;
                    this.selectedSlice = null;
                    document.getElementById('sliceViewer').style.display = 'none';
                }
                this.visualizer.selectedTags.clear();
                this.updateSelectedTagsPanel();
                this.visualizer.updateSelection();
            }
        });
    }

    async setupTagPanel() {
        // Load tags data
        const response = await fetch('/tags');
        const tagsData = await response.json();

        const tagList = document.getElementById('tagList');
        const tagSearch = document.getElementById('tagSearch');
        const selectedTagsPanel = document.getElementById('selectedTags');

        // Sort tags by size (number of items)
        const sortedTags = Object.entries(tagsData)
            .sort(([, a], [, b]) => b.itemIds.length - a.itemIds.length)
            .map(([tag]) => tag);

        // Create tag elements
        sortedTags.forEach(tag => {
            const tagElement = document.createElement('div');
            tagElement.className = 'tag-item';
            const count = tagsData[tag].itemIds.length;
            tagElement.innerHTML = `${tag} <span class="tag-count">(${count})</span>`;

            tagElement.addEventListener('click', () => {
                if (!this.visualizer.selectedTags.has(tag)) {
                    this.visualizer.selectedTags.add(tag);
                    this.updateSelectedTagsPanel();
                    this.visualizer.updateSelection();
                    tagElement.classList.add('selected');
                }
            });

            tagList.appendChild(tagElement);
        });

        // Search functionality
        tagSearch.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            Array.from(tagList.children).forEach(tagElement => {
                const tagText = tagElement.textContent.toLowerCase();
                const matches = tagText.includes(searchTerm);
                tagElement.style.display = matches ? 'block' : 'none';
            });
        });

        // Clear search on escape
        tagSearch.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                tagSearch.value = '';
                Array.from(tagList.children).forEach(tagElement => {
                    tagElement.style.display = 'block';
                });
            }
        });
    }

    updateSelectedTagsPanel() {
        const panel = document.getElementById('selectedTags');
        panel.innerHTML = '';

        this.visualizer.selectedTags.forEach(tag => {
            const tagElement = document.createElement('div');
            tagElement.className = 'selected-tag';
            tagElement.innerHTML = `${tag} <span class="remove-tag">×</span>`;

            tagElement.addEventListener('click', () => {
                this.visualizer.selectedTags.delete(tag);
                this.updateSelectedTagsPanel();
                this.visualizer.updateSelection();

                // Update tag list selection state
                const tagListItem = document.querySelector(`.tag-item:contains('${tag}')`);
                if (tagListItem) {
                    tagListItem.classList.remove('selected');
                }
            });

            panel.appendChild(tagElement);
        });
    }

    updateTooltip(event, node) {
        if (node && node.userData.hash) {
            const itemData = this.visualizer.nodeData.get(node.userData.hash);
            if (itemData) {
                // Create tooltip content
                const description = itemData.description.length > 200
                    ? itemData.description.substring(0, 200) + '...'
                    : itemData.description;

                this.tooltip.innerHTML = `
                    <div><strong>Spicy Level:</strong> ${itemData.spicy.toFixed(2)}</div>
                    <div><strong>Tags:</strong> ${Object.keys(itemData.tags).join(', ')}</div>
                    <div><strong>Description:</strong> ${description}</div>
                `;

                // Position tooltip near mouse but ensure it stays within viewport
                const tooltipWidth = this.tooltip.offsetWidth;
                const tooltipHeight = this.tooltip.offsetHeight;
                const padding = 15;

                let left = event.clientX + padding;
                let top = event.clientY + padding;

                if (left + tooltipWidth > window.innerWidth) {
                    left = event.clientX - tooltipWidth - padding;
                }
                if (top + tooltipHeight > window.innerHeight) {
                    top = event.clientY - tooltipHeight - padding;
                }

                this.tooltip.style.left = `${left}px`;
                this.tooltip.style.top = `${top}px`;
                this.tooltip.style.display = 'block';
                this.isTooltipVisible = true;
            }
        } else {
            this.hideTooltip();
        }
    }

    hideTooltip() {
        if (this.isTooltipVisible) {
            this.tooltip.style.display = 'none';
            this.isTooltipVisible = false;
        }
    }

    setupSliceViewer() {
        const viewer = document.getElementById('sliceViewer');
        const canvas = document.getElementById('sliceCanvas');
        canvas.width = 380;
        canvas.height = 380;
        this.sliceCanvas = canvas;

        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.textContent = '×';
        closeBtn.className = 'slice-close-btn';
        closeBtn.onclick = () => {
            viewer.style.display = 'none';
            if (this.selectedSlice) {
                this.selectedSlice.material.opacity = 0.1;
                this.selectedSlice = null;
            }
        };
        viewer.insertBefore(closeBtn, viewer.firstChild);
    }

    showSliceView(sliceMesh) {
        const viewer = document.getElementById('sliceViewer');
        const title = document.getElementById('sliceTitle');
        const canvas = this.sliceCanvas;
        const ctx = canvas.getContext('2d');

        // Clear canvas with semi-transparent background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw circular boundary
        ctx.beginPath();
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.arc(canvas.width/2, canvas.height/2, canvas.width/2 - 10, 0, Math.PI * 2);
        ctx.stroke();

        // Set title
        title.textContent = `Slice View - Spicy: ${sliceMesh.userData.level.toFixed(1)}`;

        // Draw nodes
        const scale = (canvas.width - 20) / (this.visualizer.SLICE_RADIUS * 2);
        const center = canvas.width / 2;

        sliceMesh.userData.items.forEach(item => {
            const node = this.visualizer.nodes.get(item.hash);
            if (!node) return;

            const x = center + node.userData.radius * Math.cos(node.userData.angle) * scale;
            const y = center + node.userData.radius * Math.sin(node.userData.angle) * scale;

            // Draw node with glow effect
            ctx.beginPath();
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
            const color = node.material.color;
            const colorStr = `rgb(${color.r * 255}, ${color.g * 255}, ${color.b * 255})`;
            gradient.addColorStop(0, colorStr);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gradient;
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();

            // Draw node center
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = colorStr;
            ctx.fill();
        });

        viewer.style.display = 'block';
    }
}